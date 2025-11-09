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
for a single college student.

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
