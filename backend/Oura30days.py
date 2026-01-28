import os
import json
import boto3
from statistics import mean
from datetime import datetime, timedelta

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET = os.environ.get("S3_BUCKET") or "healthinsights"
KEY = os.environ.get("S3_KEY") or "oura_synthetic_dataset2.json"

MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-2:406460547315:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Recommended: 7-day "current state" vs 30-day baseline
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "7"))

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "OPTIONS,GET",
}

# ---------------- dates ----------------

def parse_date_safe(d):
    # dataset is YYYY-MM-DD
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


# ---------------- load dataset ----------------

def load_records_from_s3():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    raw = obj["Body"].read().decode("utf-8")
    data = json.loads(raw)

    # Expect list of flat rows
    if not isinstance(data, list):
        raise ValueError("Expected dataset to be a JSON list of records")

    records = []
    for r in data:
        if not isinstance(r, dict):
            continue
        if "date" not in r or "user_id" not in r:
            continue
        r["_date"] = parse_date_safe(r["date"])
        records.append(r)

    records.sort(key=lambda r: r["_date"])
    return records


# ---------------- user selection + demographics ----------------

def select_user_id(records, requested_user_id=None):
    counts = {}
    for r in records:
        uid = r.get("user_id")
        if uid:
            counts[uid] = counts.get(uid, 0) + 1

    if requested_user_id and requested_user_id in counts:
        return requested_user_id

    # default: user with most rows
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None


def first_non_null(rows, key):
    for r in rows:
        v = r.get(key)
        if v is not None and v != "":
            return v
    return None


def get_user_summaries(records):
    by_user = {}
    for r in records:
        uid = r.get("user_id")
        dt = r.get("_date")
        if not uid or not dt:
            continue
        if uid not in by_user:
            by_user[uid] = {"count": 0, "min": dt, "max": dt}
        by_user[uid]["count"] += 1
        by_user[uid]["min"] = min(by_user[uid]["min"], dt)
        by_user[uid]["max"] = max(by_user[uid]["max"], dt)

    out = {}
    for uid, v in by_user.items():
        out[uid] = {
            "count": v["count"],
            "min_date": v["min"].isoformat(),
            "max_date": v["max"].isoformat(),
        }
    return out


# ---------------- numeric helpers ----------------

def sane_number(x, min_val=None, max_val=None):
    if not isinstance(x, (int, float)):
        return None
    fx = float(x)
    if min_val is not None and fx < min_val:
        return None
    if max_val is not None and fx > max_val:
        return None
    return fx


def mean_of(rows, key, min_val=None, max_val=None, round_to=2):
    vals = []
    for r in rows:
        v = sane_number(r.get(key), min_val, max_val)
        if v is not None:
            vals.append(v)
    return round(mean(vals), round_to) if vals else None


def sec_to_hours(sec):
    if not isinstance(sec, (int, float)):
        return None
    return round(float(sec) / 3600.0, 2)


# ---------------- window selection ----------------

def pick_windows(user_rows, as_of_date, window_days=7):
    """
    - Baseline: all rows strictly before as_of_date (should be ~30 days)
    - Window: last `window_days` rows/days strictly before as_of_date (assumes daily data)
    """
    baseline = [r for r in user_rows if r["_date"] < as_of_date]
    baseline.sort(key=lambda r: r["_date"])

    if not baseline:
        raise ValueError("No records before as_of_date for this user")

    # last N unique dates (robust if duplicates exist)
    unique_dates = sorted({r["_date"] for r in baseline})
    last_dates = set(unique_dates[-min(window_days, len(unique_dates)):])

    window = [r for r in baseline if r["_date"] in last_dates]
    window.sort(key=lambda r: r["_date"])

    return baseline, window, [d.isoformat() for d in sorted(last_dates)]


# ---------------- stats for flat dataset ----------------

def compute_stats(rows):
    # Adjust these bounds if needed — kept light.
    avg_sleep_hours = None
    avg_sleep_sec = mean_of(rows, "sleep_total_duration_sec", 10_800, 50_400)
    if avg_sleep_sec is not None:
        avg_sleep_hours = sec_to_hours(avg_sleep_sec)

    return {
        "num_days": len({r["_date"] for r in rows}),
        "avg_sleep_score": mean_of(rows, "sleep_score", 0, 100),
        "avg_readiness_score": mean_of(rows, "readiness_score", 0, 100),
        "avg_steps": mean_of(rows, "activity_steps", 0, 200_000),  # steps can be large
        "avg_sleep_hours": avg_sleep_hours,
        "avg_deep_hours": (sec_to_hours(mean_of(rows, "sleep_deep_duration_sec", 0, 30_000))
                           if mean_of(rows, "sleep_deep_duration_sec", 0, 30_000) is not None else None),
        "avg_rem_hours": (sec_to_hours(mean_of(rows, "sleep_rem_duration_sec", 0, 30_000))
                          if mean_of(rows, "sleep_rem_duration_sec", 0, 30_000) is not None else None),
        "avg_hrv": mean_of(rows, "average_hrv", 5, 300),
        "avg_sleep_hr": mean_of(rows, "average_heart_rate", 30, 130),
        # keep raw temp deviation average (no anomaly filtering in code)
        "avg_temp_deviation": mean_of(rows, "body_temperature_deviation", -10, 10),
    }


def build_daily_series(rows, limit=None):
    # returns daily list for Claude evidence
    series = []
    for r in rows:
        series.append({
            "date": r["_date"].isoformat(),
            "sleep_score": r.get("sleep_score"),
            "readiness_score": r.get("readiness_score"),
            "steps": r.get("activity_steps"),
            "sleep_total_duration_sec": r.get("sleep_total_duration_sec"),
            "sleep_deep_duration_sec": r.get("sleep_deep_duration_sec"),
            "sleep_rem_duration_sec": r.get("sleep_rem_duration_sec"),
            "average_heart_rate": r.get("average_heart_rate"),
            "average_hrv": r.get("average_hrv"),
            "body_temperature_deviation": r.get("body_temperature_deviation"),
            "spo2_average": r.get("spo2_average"),
            "vo2_max": r.get("vo2_max"),
        })

    if limit is not None:
        series = series[-limit:]
    return series


# ---------------- Claude prompt + call ----------------

def build_status_prompt(
    user_id,
    gender,
    age,
    as_of_date,
    window_dates,
    baseline_stats,
    window_stats,
    window_daily,
):
    return f"""
You are an expert recovery and performance coach analyzing daily wearable metrics.

This data belongs to a {gender} {age} year old.

USER:
- user_id: {user_id}
- as_of_date: {as_of_date.isoformat()}

DATA:
- We have 30 days of daily data available (baseline).
- Use the last {len(window_dates)} day(s) as the "current window".
- Window dates included (most recent): {json.dumps(window_dates)}

You are given:
- baseline_stats (30-day): {json.dumps(baseline_stats, default=str)}
- window_stats (current window): {json.dumps(window_stats, default=str)}
- window_daily (evidence): {json.dumps(window_daily, default=str)}

Important data quality rule:
- If any value is clearly implausible (e.g., extreme temperature deviation, impossible sleep durations, etc.),
  mention it as a sensor/data anomaly and do NOT use it as evidence of physiological strain.

Your job:
1) Assign EXACTLY ONE label:
  - "PerfectHealth"
  - "AboveAverage"
  - "Average"
  - "BelowAverage"
Use readiness_score + sleep_score + HRV + sleep duration primarily.

2) Reasons (2–4 bullets) with specific numbers from window_daily and/or window_stats.

3) What to do today: practical plan (training intensity, sleep target, recovery behaviors).

4) Timeline: estimate days to reach at least "AboveAverage" if they follow the plan.

Return VALID JSON only (no markdown):

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
""".strip()


def call_claude(prompt):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 900,
        "temperature": 0.4,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    })

    response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
    response_body = json.loads(response["body"].read())
    content = response_body.get("content", [])

    if isinstance(content, list):
        text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(text_blocks).strip()

    return json.dumps(response_body)


def strip_json_fences(s):
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```", "").strip()
    return t


# ---------------- handler (NO error handling) ----------------

def lambda_handler(event, context):
    params = event.get("queryStringParameters") or {}

    # CORS preflight
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod") or "GET")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    requested_user_id = params.get("user_id")
    list_users = str(params.get("list_users") or "").strip() == "1"
    date_str = params.get("date")  # optional YYYY-MM-DD

    records = load_records_from_s3()

    if list_users:
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"users": get_user_summaries(records)})
        }

    user_id = select_user_id(records, requested_user_id)
    if not user_id:
        raise ValueError("Could not determine user_id")

    user_rows = [r for r in records if r.get("user_id") == user_id]
    user_rows.sort(key=lambda r: r["_date"])

    # Capture demographics for prompt
    age = first_non_null(user_rows, "age")
    gender = first_non_null(user_rows, "gender")

    # Default as_of_date: day after the latest record for that user
    user_dates = sorted({r["_date"] for r in user_rows})
    user_min_date = user_dates[0]
    user_max_date = user_dates[-1]

    if date_str:
        as_of_date = parse_date_safe(date_str)
        if as_of_date <= user_min_date:
            raise ValueError("as_of_date must be after earliest user date")
    else:
        as_of_date = user_max_date + timedelta(days=1)

    baseline, window, window_dates = pick_windows(user_rows, as_of_date, WINDOW_DAYS)

    baseline_stats = compute_stats(baseline)
    window_stats = compute_stats(window)
    window_daily = build_daily_series(window)  # evidence for Claude

    prompt = build_status_prompt(
        user_id=user_id,
        gender=gender or "Unknown",
        age=age or "Unknown",
        as_of_date=as_of_date,
        window_dates=window_dates,
        baseline_stats=baseline_stats,
        window_stats=window_stats,
        window_daily=window_daily,
    )

    ai_plan_raw = strip_json_fences(call_claude(prompt))


    response_body = {
        "user_id": user_id,
        "as_of": as_of_date.isoformat(),
        "user_min_date": user_min_date.isoformat(),
        "user_max_date": user_max_date.isoformat(),
        "gender": gender,
        "age": age,

        "window_days": WINDOW_DAYS,
        "window_dates": window_dates,

        "baseline_stats": baseline_stats,
        "window_stats": window_stats,
        "window_daily": window_daily,

        # No parsing (since you wanted no error handling)
        "ai_plan_raw": ai_plan_raw,
    }

    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(response_body)}
