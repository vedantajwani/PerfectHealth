import os
import json
import boto3
import traceback
from statistics import mean
from datetime import datetime, timedelta

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET = os.environ.get("S3_BUCKET") or "healthinsights"
KEY = os.environ.get("S3_KEY") or "oura_synthetic_dataset.json"

MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-2:406460547315:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# "Hard max" in case you ever want more context; by default we won't use it.
CONTEXT_DAYS_BEFORE = int(os.environ.get("CONTEXT_DAYS_BEFORE", "14"))

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "OPTIONS,GET",
}


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
    raw = obj["Body"].read().decode("utf-8")
    data = json.loads(raw)

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

    return max(counts.items(), key=lambda kv: kv[1])[0]


def get_user_summaries(records):
    by_user = {}
    for r in records:
        uid = r.get("user_id")
        dt = r.get("_parsed_date")
        if not uid or not dt:
            continue
        d = dt.date()
        if uid not in by_user:
            by_user[uid] = {"count": 0, "min": d, "max": d}
        by_user[uid]["count"] += 1
        by_user[uid]["min"] = min(by_user[uid]["min"], d)
        by_user[uid]["max"] = max(by_user[uid]["max"], d)

    out = {}
    for uid, v in by_user.items():
        out[uid] = {
            "count": v["count"],
            "min_date": v["min"].isoformat(),
            "max_date": v["max"].isoformat(),
        }
    return out


# ---------------- helpers ----------------

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
    return sane_number(dot_get(r, "activity.steps"), 0, 100)


def get_sleep_total_sec(r):
    return sane_number(dot_get(r, "sleep.total_duration_sec"), 10_800, 50_400)


def get_deep_sec(r):
    return sane_number(dot_get(r, "sleep.deep_duration_sec"), 0, 30_000)


def get_rem_sec(r):
    return sane_number(dot_get(r, "sleep.rem_duration_sec"), 0, 30_000)


def get_hrv(r):
    return sane_number(dot_get(r, "sleep.avg_hrv"), 10, 200)


def get_avg_hr(r):
    return sane_number(dot_get(r, "sleep.avg_heart_rate"), 35, 120)


# ---- temp deviation: keep raw + mark status ----

def get_temp_dev_raw(r):
    x = dot_get(r, "readiness.temperature_deviation")
    return float(x) if isinstance(x, (int, float)) else None


def classify_temp_dev(temp):
    if temp is None:
        return {"value": None, "status": "missing"}
    if abs(temp) <= 3.0:
        return {"value": float(temp), "status": "normal"}
    return {"value": float(temp), "status": "abnormal"}


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


# ---------------- recent-day selection (NEW) ----------------

def select_most_recent_unique_days(history_records, as_of_date, desired_unique_days=2, hard_max_unique_days=14):
    """
    Returns rows covering the last N *unique dates* prior to as_of_date.
    - Primary behavior: N=2 (compare to day before)
    - If only 1 day exists, returns that 1 day.
    - If there are gaps, still returns the last 2 available days.
    - hard_max_unique_days caps how many days we could ever return.
    """
    eligible = [r for r in history_records if r.get("_parsed_date") and r["_parsed_date"].date() < as_of_date]
    if not eligible:
        return [], []

    unique_dates = sorted({r["_parsed_date"].date() for r in eligible})
    # how many do we actually use?
    k = min(desired_unique_days, len(unique_dates))
    k = max(1, k)

    # just in case: cap by hard max
    k = min(k, max(1, hard_max_unique_days))

    chosen_dates = unique_dates[-k:]
    chosen_rows = [r for r in eligible if r["_parsed_date"].date() in set(chosen_dates)]
    chosen_rows.sort(key=lambda r: r["_parsed_date"])

    return chosen_rows, [d.isoformat() for d in chosen_dates]


# ---------------- Oura stats ----------------

def compute_stats_oura(records):
    if not records:
        return {"num_days": 0}

    unique_days = len({r["_parsed_date"].date() for r in records if r.get("_parsed_date")})

    avg_sleep_sec = mean_of(records, get_sleep_total_sec)
    avg_sleep_h = sec_to_hours(avg_sleep_sec) if avg_sleep_sec is not None else None

    # avg temp deviation from NORMAL temps only
    normal_temps = []
    for r in records:
        t = get_temp_dev_raw(r)
        if isinstance(t, (int, float)) and abs(t) <= 3.0:
            normal_temps.append(float(t))
    avg_temp = round(mean(normal_temps), 2) if normal_temps else None

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
        "avg_temp_deviation_c": avg_temp,
    }


def build_recent_daily_summary_from_rows(rows):
    out = []
    for r in rows:
        temp_info = classify_temp_dev(get_temp_dev_raw(r))
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
            "temp_deviation_c": temp_info["value"],
            "temp_deviation_status": temp_info["status"],
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

    def stat_temp_normal():
        vals = []
        for r in day_rows:
            t = get_temp_dev_raw(r)
            if isinstance(t, (int, float)) and abs(t) <= 3.0:
                vals.append(float(t))
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
        "temp_deviation_c": stat_temp_normal(),
    }


# ---------------- Claude prompt ----------------

def build_status_prompt_oura(
    user_id,
    as_of_date,
    overall_stats,
    recent_stats,
    recent_daily,
    history_days,
    recent_days_used,
    recent_dates,
    cohort_snapshot,
):
    return f"""
You are an expert recovery and performance coach analyzing Oura Ring-style daily metrics.

USER:
- user_id: {user_id}
- as_of_date: {as_of_date.isoformat()}

DATA LIMITS:
- Total days available for this user: {history_days}
- Recent comparison used: {recent_days_used} day(s)
- Recent dates included (most recent): {json.dumps(recent_dates)}

IMPORTANT:
Default behavior is day-over-day comparison (latest day vs previous day).
If only one day exists, do not claim trends; only describe that single day.

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
- temp_deviation_c is temperature deviation in °C.
- temp_deviation_status can be:
    - "normal" (|temp| <= 3.0)
    - "abnormal" (not physiologically plausible in this dataset; treat as data abnoramlly)
    - "missing"

IMPORTANT DATA QUALITY RULE:
If temp_deviation_status is "abnormal", you MUST mention it as a data/sensor/ingestion anomaly
and you MUST NOT use it as evidence of physiological strain.

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
If any metric is flagged abnormal, mention the anomaly and say you did not rely on it.

3) WHAT TO DO TODAY
Give a practical plan for today (sleep hours, easy vs hard physical activity, recovery focus).
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

        # Preflight
        method = (event.get("requestContext", {}).get("http", {}).get("method")
                  or event.get("httpMethod") or "GET")
        if method == "OPTIONS":
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

        date_str = params.get("date")
        requested_user_id = params.get("user_id")
        list_users = str(params.get("list_users") or "").strip() == "1"

        records = load_records_from_s3()
        dated = [r for r in records if r.get("_parsed_date") is not None]
        if not dated:
            raise ValueError("No dated records found in dataset")

        if list_users:
            summaries = get_user_summaries(dated)
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"users": summaries})}

        # Pick user
        user_id = select_user_id(dated, requested_user_id)
        if not user_id:
            raise ValueError("Could not determine user_id for analysis")

        user_rows = [r for r in dated if r.get("user_id") == user_id]
        user_rows.sort(key=lambda r: r["_parsed_date"])

        user_dates = sorted({r["_parsed_date"].date() for r in user_rows if r.get("_parsed_date")})
        if not user_dates:
            raise ValueError(f"No dated records for user_id={user_id}")

        user_min_date = user_dates[0]
        user_max_date = user_dates[-1]

        # as_of_date: default latest for this user + 1
        if date_str:
            as_of_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if as_of_date <= user_min_date:
                raise ValueError(f"as_of_date {as_of_date} is not after earliest user data date {user_min_date}")
        else:
            as_of_date = user_max_date + timedelta(days=1)

        # History before as_of_date
        history = [r for r in user_rows if r["_parsed_date"].date() < as_of_date]
        if not history:
            raise ValueError("No historical records before the requested date for this user")

        history_days = len({r["_parsed_date"].date() for r in history})

        # ✅ NEW: pick only most recent 2 unique days for comparisons
        recent_rows, recent_dates = select_most_recent_unique_days(
            history_records=history,
            as_of_date=as_of_date,
            desired_unique_days=2,                 # day-over-day
            hard_max_unique_days=CONTEXT_DAYS_BEFORE
        )
        recent_days_used = len(set([r["_parsed_date"].date() for r in recent_rows])) if recent_rows else 0

        overall_stats = compute_stats_oura(history)
        recent_stats = compute_stats_oura(recent_rows) if recent_rows else {}

        recent_daily = build_recent_daily_summary_from_rows(recent_rows)
        cohort_snapshot = compute_cohort_snapshot(dated, as_of_date)

        prompt = build_status_prompt_oura(
            user_id=user_id,
            as_of_date=as_of_date,
            overall_stats=overall_stats,
            recent_stats=recent_stats,
            recent_daily=recent_daily,
            history_days=history_days,
            recent_days_used=recent_days_used,
            recent_dates=recent_dates,
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
            "user_min_date": user_min_date.isoformat(),
            "user_max_date": user_max_date.isoformat(),
            "history_days": history_days,

            # ✅ NEW: explicitly return what we used for comparisons
            "recent_days_used": recent_days_used,
            "recent_dates": recent_dates,

            "overall_stats": overall_stats,
            "recent_stats": recent_stats,
            "recent_daily": recent_daily,
            "cohort_snapshot": cohort_snapshot,
            "ai_plan": ai_plan,
        }

        return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(response_body)}

    except Exception as e:
        trace = traceback.format_exc()
        print(trace)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e), "trace": trace}),
        }
