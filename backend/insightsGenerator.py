import os
import json
import boto3
import traceback
from statistics import mean
from datetime import datetime, timedelta

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

# Environment-driven config (safe for GitHub)
BUCKET = os.environ.get("S3_BUCKET", "your-s3-bucket-name-here")
KEY = os.environ.get("S3_KEY", "final_dataset.json")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    # Example Bedrock Claude model id (no account id exposed)
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
)

CONTEXT_DAYS_BEFORE = int(os.environ.get("CONTEXT_DAYS_BEFORE", "14"))
CONTEXT_DAYS_AFTER = int(os.environ.get("CONTEXT_DAYS_AFTER", "7"))


# ---------- helpers for dates & stats ----------

def parse_date_safe(d):
    if not d:
        return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def load_records_from_s3():
    """
    Load health dataset JSON from S3 and attach _parsed_date.
    Skips any records that don't have a usable date.
    """
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    body = obj["Body"].read().decode("utf-8")
    data = json.loads(body)

    # Support either a list or {"records": [...]}
    if isinstance(data, dict) and "records" in data:
        raw_records = data["records"]
    elif isinstance(data, list):
        raw_records = data
    else:
        raise ValueError("Unexpected JSON structure for dataset")

    records = []
    for r in raw_records:
        raw_date = r.get("date") or r.get("day")
        if not raw_date:
            continue

        parsed = parse_date_safe(raw_date)
        if parsed is None:
            continue

        r["_parsed_date"] = parsed
        records.append(r)

    if not records:
        raise ValueError("No valid dated records found in dataset")

    records.sort(key=lambda r: r["_parsed_date"])
    return records


def safe_mean(records, key):
    vals = [
        r[key] for r in records
        if key in r and isinstance(r[key], (int, float))
    ]
    return round(mean(vals), 2) if vals else None


def compute_stats(records):
    if not records:
        return {"num_days": 0}

    stats = {
        "num_days": len(records),
        "avg_sleep_hours": safe_mean(records, "sleep_hours_total")
                           or safe_mean(records, "sleep_hours"),
        "avg_sleep_efficiency_pct": safe_mean(records, "sleep_efficiency_pct"),
        "avg_sleep_performance_pct": safe_mean(records, "sleep_performance_pct"),
        "avg_recovery_score_pct": safe_mean(records, "recovery_score_pct"),
        "avg_day_strain": safe_mean(records, "day_strain"),
        "avg_resting_hr_bpm": safe_mean(records, "resting_hr_bpm"),
        "avg_hrv_ms": safe_mean(records, "hrv_ms"),
        "avg_skin_temp_c": safe_mean(records, "skin_temp_c"),
        "avg_blood_oxygen_pct": safe_mean(records, "blood_oxygen_pct"),
        "avg_caffeine_mg": safe_mean(records, "caffeine_mg"),
        "alcohol_nights": sum(
            1 for r in records
            if str(r.get("alcohol", 0)).lower() in ("1", "true", "yes")
        ),
    }
    return stats


def compute_recovery_transition_stats(records):
    days = [
        r for r in records
        if r.get("_parsed_date") is not None
        and r.get("recovery_score_pct") is not None
    ]
    if not days:
        return {"num_events": 0}

    days = sorted(days, key=lambda r: r["_parsed_date"])

    def bucket(recovery):
        if recovery is None:
            return None
        if recovery < 33:
            return "low"
        if recovery < 67:
            return "medium"
        return "high"

    transitions = []
    n = len(days)

    for i, r in enumerate(days):
        if bucket(r["recovery_score_pct"]) == "low":
            start_date = r["_parsed_date"].date()
            for j in range(i + 1, n):
                if bucket(days[j]["recovery_score_pct"]) == "high":
                    end_date = days[j]["_parsed_date"].date()
                    delta = (end_date - start_date).days
                    if delta >= 0:
                        transitions.append(delta)
                    break

    if not transitions:
        return {"num_events": 0}

    avg_days = sum(transitions) / len(transitions)
    return {
        "num_events": len(transitions),
        "min_days_low_to_high": min(transitions),
        "max_days_low_to_high": max(transitions),
        "avg_days_low_to_high": avg_days,
        "all_durations_days": transitions,
    }


def build_recent_daily_summary(history_records, as_of_date, window_days=14):
    window_start = as_of_date - timedelta(days=window_days)

    dated = [
        r for r in history_records
        if r.get("_parsed_date") is not None
        and window_start <= r["_parsed_date"].date() < as_of_date
    ]

    by_day = {}
    for r in dated:
        d = r["_parsed_date"].date().isoformat()
        by_day.setdefault(d, []).append(r)

    daily_summary = []
    for d, rows in sorted(by_day.items()):
        main = max(
            rows,
            key=lambda x: (x.get("recovery_score_pct") is not None, x.get("recovery_score_pct") or 0)
        )

        total_sleep_hours = sum(
            (row.get("sleep_hours") or 0) for row in rows
        )

        daily_summary.append({
            "date": d,
            "recovery_score_pct": main.get("recovery_score_pct"),
            "resting_hr_bpm": main.get("resting_hr_bpm"),
            "hrv_ms": main.get("hrv_ms"),
            "skin_temp_c": main.get("skin_temp_c"),
            "day_strain": main.get("day_strain"),
            "sleep_hours_total": total_sleep_hours or main.get("sleep_hours"),
            "sleep_efficiency_pct": main.get("sleep_efficiency_pct"),
            "sleep_debt_min": main.get("sleep_debt_min"),
            "sleep_need_min": main.get("sleep_need_min"),
            "had_alcohol": bool(main.get("had_alcohol")),
            "had_caffeine": bool(main.get("had_caffeine")),
            "ate_late": bool(main.get("ate_late")),
            "used_marijuana": bool(main.get("used_marijuana")),
            "used_melatonin": bool(main.get("used_melatonin")),
            "nap": main.get("nap"),
            "total_workout_minutes": main.get("total_workout_minutes"),
            "avg_activity_strain": main.get("avg_activity_strain"),
            "total_busy_minutes": main.get("total_busy_minutes"),
            "has_exam": bool(main.get("has_exam")),
            "has_task_due": bool(main.get("has_task_due")),
            "has_workout": bool(main.get("has_workout")),
            "has_long_tasks": bool(main.get("has_long_tasks")),
        })

    return daily_summary


def compute_event_summary(records):
    if not records:
        return {}

    day_set = {
        r["_parsed_date"].date()
        for r in records
        if r.get("_parsed_date") is not None
    }
    n_days = len(day_set) or 1

    def bool_rate(key):
        vals = [
            bool(r.get(key))
            for r in records
            if r.get("_parsed_date") is not None and key in r
        ]
        if not vals:
            return {"count_days": 0, "pct_days": 0.0}
        count = sum(vals)
        return {
            "count_days": int(count),
            "pct_days": (count / n_days) * 100.0,
        }

    return {
        "num_days_with_data": n_days,
        "alcohol": bool_rate("had_alcohol"),
        "caffeine": bool_rate("had_caffeine"),
        "ate_late": bool_rate("ate_late"),
        "marijuana": bool_rate("used_marijuana"),
        "melatonin": bool_rate("used_melatonin"),
        "has_exam": bool_rate("has_exam"),
        "has_task_due": bool_rate("has_task_due"),
        "has_workout_flag": bool_rate("has_workout"),
        "has_long_tasks": bool_rate("has_long_tasks"),
    }


def compute_daily_risk(selected_records, baseline_records):
    if not selected_records:
        return {"overall_risk": "unknown", "daily": []}

    base_recovery = safe_mean(baseline_records, "recovery_score_pct")
    base_hrv = safe_mean(baseline_records, "hrv_ms")
    base_rest_hr = safe_mean(baseline_records, "resting_hr_bpm")
    base_temp = safe_mean(baseline_records, "skin_temp_c")
    base_caffeine = safe_mean(baseline_records, "caffeine_mg") or 0.0

    daily = []
    levels_map = []

    for r in selected_records:
        d = r["_parsed_date"].date().isoformat() if r.get("_parsed_date") else str(r.get("date"))
        score = 0.0
        reasons = []

        recovery = r.get("recovery_score_pct")
        if isinstance(recovery, (int, float)) and base_recovery is not None:
            if recovery <= base_recovery - 20:
                score += 2.0
                reasons.append("recovery much lower than usual")
            elif recovery <= base_recovery - 10:
                score += 1.0
                reasons.append("recovery slightly lower than usual")

        hrv = r.get("hrv_ms")
        if isinstance(hrv, (int, float)) and base_hrv is not None:
            if hrv <= base_hrv * 0.7:
                score += 1.5
                reasons.append("HRV significantly lower than baseline")
            elif hrv <= base_hrv * 0.85:
                score += 0.75
                reasons.append("HRV a bit lower than baseline")

        rest_hr = r.get("resting_hr_bpm")
        if isinstance(rest_hr, (int, float)) and base_rest_hr is not None:
            if rest_hr >= base_rest_hr + 8:
                score += 1.5
                reasons.append("resting HR significantly elevated")
            elif rest_hr >= base_rest_hr + 4:
                score += 0.75
                reasons.append("resting HR slightly elevated")

        temp = r.get("skin_temp_c")
        if isinstance(temp, (int, float)) and base_temp is not None:
            if temp >= base_temp + 0.5:
                score += 1.5
                reasons.append("skin temperature elevated")
            elif temp >= base_temp + 0.3:
                score += 0.75
                reasons.append("skin temperature slightly elevated")

        alcohol = str(r.get("alcohol", 0)).lower() in ("1", "true", "yes")
        if alcohol:
            score += 0.75
            reasons.append("alcohol consumption")

        caffeine = r.get("caffeine_mg") or 0.0
        if isinstance(caffeine, (int, float)) and caffeine >= base_caffeine + 100:
            score += 0.5
            reasons.append("unusually high caffeine")

        if score >= 3.0:
            level = "high"
        elif score >= 1.5:
            level = "medium"
        else:
            level = "low"

        levels_map.append(level)
        daily.append({
            "date": d,
            "score": round(score, 2),
            "level": level,
            "reasons": reasons,
        })

    if "high" in levels_map:
        overall = "high"
    elif "medium" in levels_map:
        overall = "medium"
    else:
        overall = "low"

    return {
        "overall_risk": overall,
        "daily": daily,
    }


def build_status_prompt(
    as_of_date,
    overall_stats,
    recent_14d_stats,
    recent_7d_stats,
    event_summary,
    recent_daily,
    recovery_transition_stats,
):
    return f"""
You are an expert recovery and performance coach analyzing data from a wearable and life log
for a male 21 year old.

Today (the date the user cares about) is: {as_of_date.isoformat()}.

You are given:
- overall_stats: {json.dumps(overall_stats, default=str)}
- recent_14d_stats (last 14 days before today): {json.dumps(recent_14d_stats, default=str)}
- recent_7d_stats (last 7 days before today): {json.dumps(recent_7d_stats, default=str)}
- event_summary (frequency of alcohol, caffeine, exams, long task days, etc.): {json.dumps(event_summary, default=str)}
- recent_daily (last ~14 days of day-level metrics including recovery, sleep, strain, and habits): {json.dumps(recent_daily, default=str)}
- recovery_transition_stats (how long it historically takes to go from low to high recovery): {json.dumps(recovery_transition_stats, default=str)}

DEFINITIONS:
- "Recovery" is recovery_score_pct (0–100). High is good.
- "Strain" is day_strain (higher = more load).
- "Baseline" means the user's long-term averages in overall_stats.
- "Recent" means the last 7–14 days in recent_stats and recent_daily.

YOUR JOB:

1) BODY CONDITION
Assign the user's current body condition to EXACTLY ONE of these labels:
  - "PerfectHealth"
  - "AboveAverage"
  - "Average"
  - "BelowAverage"

Base this on:
  - How recent recovery compares to their own baseline.
  - How resting HR, HRV, and skin temperature look vs baseline.
  - How much sleep they are actually getting vs sleep_need_min and sleep_debt_min.
  - How heavy or light their recent strain has been.

2) REASONS (DATA-BACKED)
Explain WHY you chose that label in 2–4 concise bullet points, each referencing specific data patterns.

3) WHAT TO DO TODAY
Give a specific plan JUST FOR TODAY that helps move them toward "PerfectHealth".

4) HOW LONG TO IMPROVE
Estimate how many days it will PROBABLY take them to reach at least "AboveAverage" body condition
if they follow your advice, using their own history instead of generic averages.

FORMAT:

Return your answer as VALID JSON (no markdown fences) in exactly this shape:

{{
  "condition_label": "PerfectHealth | AboveAverage | Average | BelowAverage",
  "condition_summary": "2-3 lines about how their body is doing right now",
  "reasons": [
    {{"factor": "Sleep", "evidence": "Example: 'Last 7 days: 5.9 h vs baseline 7.6 h; sleep debt is elevated at ~90 min.'"}},
    {{"factor": "Strain", "evidence": "Example: 'Strain has averaged 15.2 vs your usual 11.0, including one spike around 20.'" }}
  ],
  "today_plan": "bullet-style text describing exactly what they should do today (workout vs rest, bedtime, naps, caffeine, alcohol, melatonin).",
  "estimated_improvement_days": "a short string range like '2-3' or '3-5'",
  "improvement_rationale": "one short paragraph explaining the timeline using their past recovery patterns."
}}

Only return JSON. Do not include any explanation outside the JSON object.
"""


def call_claude(prompt):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1200,
            "temperature": 0.4,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
            ],
        }
    )

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=body
    )

    response_body = json.loads(response["body"].read())
    content = response_body.get("content", [])
    if content and isinstance(content, list):
        text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(text_blocks).strip()
    return json.dumps(response_body)


def lambda_handler(event, context):
    try:
        params = event.get("queryStringParameters") or {}
        date_str = params.get("date")

        records = load_records_from_s3()
        dated = [r for r in records if r.get("_parsed_date") is not None]

        if not dated:
            raise ValueError("No dated records found in dataset")

        all_dates = sorted(r["_parsed_date"].date() for r in dated)
        min_date = all_dates[0]
        max_date = all_dates[-1]

        if date_str:
            as_of_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            as_of_date = max_date + timedelta(days=1)

        if as_of_date <= min_date:
            raise ValueError(
                f"as_of_date {as_of_date} is not after the earliest data date {min_date}"
            )

        history_records = [
            r for r in dated if r["_parsed_date"].date() < as_of_date
        ]

        if not history_records:
            raise ValueError("No historical records before the requested date")

        overall_stats = compute_stats(history_records)

        def filter_window(days_back):
            start = as_of_date - timedelta(days=days_back)
            return [
                r for r in history_records
                if start <= r["_parsed_date"].date() < as_of_date
            ]

        recent_14 = filter_window(14)
        recent_7 = filter_window(7)

        recent_14d_stats = compute_stats(recent_14) if recent_14 else {}
        recent_7d_stats = compute_stats(recent_7) if recent_7 else {}

        event_summary = compute_event_summary(history_records)
        recent_daily = build_recent_daily_summary(history_records, as_of_date, window_days=14)
        recovery_transition_stats = compute_recovery_transition_stats(history_records)

        prompt = build_status_prompt(
            as_of_date=as_of_date,
            overall_stats=overall_stats,
            recent_14d_stats=recent_14d_stats,
            recent_7d_stats=recent_7d_stats,
            event_summary=event_summary,
            recent_daily=recent_daily,
            recovery_transition_stats=recovery_transition_stats,
        )

        raw_output = call_claude(prompt)

        try:
            ai_plan = json.loads(raw_output)
        except json.JSONDecodeError:
            ai_plan = {
                "parse_error": "Could not parse Claude output as JSON",
                "raw_text": raw_output,
            }

        response_body = {
            "as_of": as_of_date.isoformat(),
            "overall_stats": overall_stats,
            "recent_14d_stats": recent_14d_stats,
            "recent_7d_stats": recent_7d_stats,
            "event_summary": event_summary,
            "recent_daily": recent_daily,
            "recovery_transition_stats": recovery_transition_stats,
            "ai_plan": ai_plan,
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,GET",
            },
            "body": json.dumps(response_body),
        }

    except Exception as e:
        trace = traceback.format_exc()
        print(trace)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,GET",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e), "trace": trace}),
        }
import os
import json
import boto3
import traceback
from statistics import mean
from datetime import datetime, timedelta

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET = os.environ.get("S3_BUCKET") or "healthinsights"
KEY = os.environ.get("S3_KEY") or "oura_dataset.json"
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-2:406460547315:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# These are "desired" windows; we'll clamp to available days per user
CONTEXT_DAYS_BEFORE = int(os.environ.get("CONTEXT_DAYS_BEFORE", "14"))


# ---------------- dates ----------------

def parse_date_safe(d):
    if not d:
        return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def load_records_from_s3():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    if isinstance(data, dict) and "records" in data:
        raw_records = data["records"]
    elif isinstance(data, list):
        raw_records = data
    else:
        raise ValueError("Unexpected JSON structure for dataset (expected list or {'records': [...]})")

    records = []
    for r in raw_records:
        raw_date = r.get("date") or r.get("day")
        parsed = parse_date_safe(raw_date)
        if not parsed:
            continue
        r["_parsed_date"] = parsed
        records.append(r)

    if not records:
        raise ValueError("No valid dated records found in dataset")

    records.sort(key=lambda r: r["_parsed_date"])
    return records


# ---------------- multi-user selection ----------------

def select_user_id(records, requested_user_id=None):
    counts = {}
    for r in records:
        uid = r.get("user_id")
        if uid and r.get("_parsed_date"):
            counts[uid] = counts.get(uid, 0) + 1

    if not counts:
        return None

    if requested_user_id and requested_user_id in counts:
        return requested_user_id

    # Default: pick the user with the most rows (best for demo)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def clamp_window_days(history_records, requested_days):
    unique_days = sorted({r["_parsed_date"].date() for r in history_records if r.get("_parsed_date")})
    available = len(unique_days)
    if available <= 0:
        return 0
    return max(1, min(requested_days, available))


# ---------------- safe extraction helpers ----------------

def dot_get(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def sane_number(x, min_val=None, max_val=None):
    if not isinstance(x, (int, float)):
        return None
    if min_val is not None and x < min_val:
        return None
    if max_val is not None and x > max_val:
        return None
    return float(x)


def get_sleep_score(r):
    return sane_number(dot_get(r, "sleep.score"), 0, 100)


def get_readiness_score(r):
    return sane_number(dot_get(r, "readiness.score"), 0, 100)


def get_activity_steps(r):
    # Your synthetic steps look normalized (60-90), not real step counts.
    # We'll treat it as a relative "steps index" 0-100.
    return sane_number(dot_get(r, "activity.steps"), 0, 100)


def get_sleep_total_sec(r):
    # Typical 3h–14h => 10800–50400 sec
    return sane_number(dot_get(r, "sleep.total_duration_sec"), 10_800, 50_400)


def get_deep_sec(r):
    return sane_number(dot_get(r, "sleep.deep_duration_sec"), 0, 30_000)


def get_rem_sec(r):
    return sane_number(dot_get(r, "sleep.rem_duration_sec"), 0, 30_000)


def get_hrv(r):
    # Typical HRV 10–200ms
    return sane_number(dot_get(r, "sleep.avg_hrv"), 10, 200)


def get_avg_hr(r):
    # Typical avg HR during sleep 35–120 bpm
    return sane_number(dot_get(r, "sleep.avg_heart_rate"), 35, 120)


def get_temp_dev(r):
    # Temperature deviation is typically small, like -2.0 to +2.0 C
    # Your dataset has a few broken values (9000 etc). Filter those out.
    return sane_number(dot_get(r, "readiness.temperature_deviation"), -3.0, 3.0)


def mean_of(records, fn):
    vals = []
    for r in records:
        v = fn(r)
        if isinstance(v, (int, float)):
            vals.append(v)
    return round(mean(vals), 2) if vals else None


def sec_to_hours(sec):
    if not isinstance(sec, (int, float)):
        return None
    return round(sec / 3600.0, 2)


# ---------------- Oura stats ----------------

def compute_stats_oura(records):
    if not records:
        return {"num_days": 0}

    unique_days = len({r["_parsed_date"].date() for r in records if r.get("_parsed_date")})

    avg_sleep_sec = mean_of(records, get_sleep_total_sec)
    avg_sleep_h = sec_to_hours(avg_sleep_sec) if avg_sleep_sec is not None else None

    return {
        "num_days": unique_days,
        "avg_sleep_score": mean_of(records, get_sleep_score),
        "avg_readiness_score": mean_of(records, get_readiness_score),
        "avg_steps_index": mean_of(records, get_activity_steps),

        "avg_sleep_hours": avg_sleep_h,
        "avg_deep_hours": sec_to_hours(mean_of(records, get_deep_sec)),
        "avg_rem_hours": sec_to_hours(mean_of(records, get_rem_sec)),

        "avg_hrv_ms": mean_of(records, get_hrv),
        "avg_sleep_heart_rate_bpm": mean_of(records, get_avg_hr),
        "avg_temp_deviation_c": mean_of(records, get_temp_dev),
    }


def build_recent_daily_summary_oura(history_records, as_of_date, window_days):
    if window_days <= 0:
        return []

    window_start = as_of_date - timedelta(days=window_days)
    rows = [
        r for r in history_records
        if r.get("_parsed_date") and window_start <= r["_parsed_date"].date() < as_of_date
    ]

    rows.sort(key=lambda r: r["_parsed_date"])

    out = []
    for r in rows:
        out.append({
            "date": r["_parsed_date"].date().isoformat(),
            "sleep_score": get_sleep_score(r),
            "readiness_score": get_readiness_score(r),
            "steps_index": get_activity_steps(r),
            "sleep_hours": sec_to_hours(get_sleep_total_sec(r)),
            "deep_hours": sec_to_hours(get_deep_sec(r)),
            "rem_hours": sec_to_hours(get_rem_sec(r)),
            "hrv_ms": get_hrv(r),
            "avg_sleep_hr_bpm": get_avg_hr(r),
            "temp_deviation_c": get_temp_dev(r),
        })
    return out


def compute_cohort_snapshot(records_all_users, as_of_date):
    all_days = sorted({r["_parsed_date"].date() for r in records_all_users if r.get("_parsed_date")})
    if not all_days:
        return {}

    latest_day = None
    for d in reversed(all_days):
        if d < as_of_date:
            latest_day = d
            break
    if latest_day is None:
        return {}

    day_rows = [r for r in records_all_users if r.get("_parsed_date") and r["_parsed_date"].date() == latest_day]

    def stat(fn):
        vals = [fn(r) for r in day_rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        vals.sort()
        return {"count": len(vals), "median": vals[len(vals)//2], "min": vals[0], "max": vals[-1]}

    return {
        "cohort_day": latest_day.isoformat(),
        "sleep_score": stat(get_sleep_score),
        "readiness_score": stat(get_readiness_score),
        "steps_index": stat(get_activity_steps),
        "hrv_ms": stat(get_hrv),
        "avg_sleep_hr_bpm": stat(get_avg_hr),
        "temp_deviation_c": stat(get_temp_dev),
    }


# ---------------- Claude prompt ----------------

def build_status_prompt_oura(
    user_id,
    as_of_date,
    overall_stats,
    recent_stats,
    recent_daily,
    history_days,
    window_days,
    cohort_snapshot,
):
    return f"""
You are an expert recovery and performance coach analyzing Oura Ring-style daily metrics.

USER:
- user_id: {user_id}
- as_of_date: {as_of_date.isoformat()}

DATA LIMITS:
- Total days available for this user: {history_days}
- Recent window used: {window_days} days (clamped to available data).
If days < 7, do NOT claim "weekly trends". Focus on day-to-day changes and cohort comparison.

You are given:
- overall_stats: {json.dumps(overall_stats, default=str)}
- recent_stats: {json.dumps(recent_stats, default=str)}
- recent_daily: {json.dumps(recent_daily, default=str)}
- cohort_snapshot: {json.dumps(cohort_snapshot, default=str)}

Definitions:
- sleep_score and readiness_score are 0–100 (higher is better).
- steps_index is a 0–100 relative index (not literal steps).
- HRV is in ms (higher generally better).
- avg_sleep_heart_rate_bpm is sleeping heart rate (lower can be better, context matters).
- avg_temp_deviation_c is filtered; missing means "not reliable" in this dataset.

YOUR JOB:

1) BODY CONDITION
Assign EXACTLY ONE label:
  - "PerfectHealth"
  - "AboveAverage"
  - "Average"
  - "BelowAverage"

Base it on readiness_score + sleep_score + HRV + sleep hours, and optionally compare to cohort.

2) REASONS (DATA-BACKED)
Give 2–4 concise bullet points referencing specific numbers from recent_daily / stats.

3) WHAT TO DO TODAY
Give a practical plan for today (sleep timing, easy vs hard training, recovery focus).
Do not invent information we do not have.

4) TIMELINE
Estimate how many days to reach at least "AboveAverage" if they follow the plan.
If history is tiny, say it’s a rough estimate.

FORMAT:
Return VALID JSON (no markdown fences) exactly:

{{
  "condition_label": "PerfectHealth | AboveAverage | Average | BelowAverage",
  "condition_summary": "2-3 lines",
  "reasons": [
    {{"factor": "Sleep", "evidence": "..." }},
    {{"factor": "Readiness", "evidence": "..." }}
  ],
  "today_plan": "bullet-style text",
  "estimated_improvement_days": "e.g. '1-3' or '3-5'",
  "improvement_rationale": "one short paragraph"
}}

Only return JSON.
"""


def call_claude(prompt):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1200,
            "temperature": 0.4,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    )
    response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
    response_body = json.loads(response["body"].read())
    content = response_body.get("content", [])
    if content and isinstance(content, list):
        text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(text_blocks).strip()
    return json.dumps(response_body)


# ---------------- handler ----------------

def lambda_handler(event, context):
    try:
        params = event.get("queryStringParameters") or {}
        date_str = params.get("date")
        requested_user_id = params.get("user_id")

        records = load_records_from_s3()
        dated = [r for r in records if r.get("_parsed_date") is not None]
        if not dated:
            raise ValueError("No dated records found in dataset")

        # Determine as_of_date
        all_dates = sorted(r["_parsed_date"].date() for r in dated)
        min_date = all_dates[0]
        max_date = all_dates[-1]

        if date_str:
            as_of_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            as_of_date = max_date + timedelta(days=1)

        if as_of_date <= min_date:
            raise ValueError(f"as_of_date {as_of_date} is not after earliest data date {min_date}")

        # Pick user
        user_id = select_user_id(dated, requested_user_id)
        if not user_id:
            raise ValueError("Could not determine user_id for analysis")

        user_rows = [r for r in dated if r.get("user_id") == user_id]
        user_rows.sort(key=lambda r: r["_parsed_date"])

        # History strictly before as_of_date
        history = [r for r in user_rows if r["_parsed_date"].date() < as_of_date]
        if not history:
            raise ValueError("No historical records before the requested date for this user")

        history_days = len({r["_parsed_date"].date() for r in history})

        # Clamp windows
        window_days = clamp_window_days(history, CONTEXT_DAYS_BEFORE)

        def filter_window(days_back):
            start = as_of_date - timedelta(days=days_back)
            return [r for r in history if start <= r["_parsed_date"].date() < as_of_date]

        recent = filter_window(window_days)

        overall_stats = compute_stats_oura(history)
        recent_stats = compute_stats_oura(recent) if recent else {}

        recent_daily = build_recent_daily_summary_oura(history, as_of_date, window_days=window_days)
        cohort_snapshot = compute_cohort_snapshot(dated, as_of_date)

        prompt = build_status_prompt_oura(
            user_id=user_id,
            as_of_date=as_of_date,
            overall_stats=overall_stats,
            recent_stats=recent_stats,
            recent_daily=recent_daily,
            history_days=history_days,
            window_days=window_days,
            cohort_snapshot=cohort_snapshot,
        )

        raw_output = call_claude(prompt)

        try:
            ai_plan = json.loads(raw_output)
        except json.JSONDecodeError:
            ai_plan = {"parse_error": "Could not parse Claude output as JSON", "raw_text": raw_output}

        response_body = {
            "user_id": user_id,
            "as_of": as_of_date.isoformat(),
            "history_days": history_days,
            "window_days": window_days,
            "overall_stats": overall_stats,
            "recent_stats": recent_stats,
            "recent_daily": recent_daily,
            "cohort_snapshot": cohort_snapshot,
            "ai_plan": ai_plan,
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,GET",
            },
            "body": json.dumps(response_body),
        }

    except Exception as e:
        trace = traceback.format_exc()
        print(trace)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e), "trace": trace}),
        }
