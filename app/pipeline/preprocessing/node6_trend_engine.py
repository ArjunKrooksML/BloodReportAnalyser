import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.db.store import get_patient_history
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512)

INTERPRET_PROMPT = """You are a clinical analyst interpreting a lab marker trend over time.

Marker: {name} ({unit})
Historical readings (oldest to newest): {history}
Current reading: {current} {unit}
Reference range: {ref_low} – {ref_high}

Computed stats:
- Change from previous: {change_percent:+.1f}%
- Monthly rate: {monthly_rate}
- Projected threshold crossing: {months_to_threshold}

Tasks:
1. Classify the trajectory as exactly one of: IMPROVING, STABLE, WORSENING, CRITICAL_TREND
   Use your clinical knowledge about this specific marker — not just the percentage.
   - IMPROVING: moving toward or within normal range in a meaningful way
   - STABLE: minimal change with no clinical concern
   - WORSENING: trending away from normal in a clinically significant way
   - CRITICAL_TREND: rapid deterioration or dangerous level with continued decline

2. Write one sentence of clinical interpretation. Be specific about what this rate of change means.
   Do not start with "The". Start with the marker name or a clinical observation.

Respond with JSON only:
{{"trajectory": "IMPROVING|STABLE|WORSENING|CRITICAL_TREND", "interpretation": "..."}}"""

VALID_TRAJECTORIES = {"IMPROVING", "STABLE", "WORSENING", "CRITICAL_TREND"}


def _find_marker_in_history(marker_name: str, history_sessions: list[dict]) -> list[dict]:
    name_lower = marker_name.lower()
    readings = []
    for session in history_sessions:
        for m in session["markers"]:
            if name_lower in m.get("name", "").lower() or m.get("name", "").lower() in name_lower:
                readings.append({
                    "value": m.get("value"),
                    "unit": m.get("unit", ""),
                    "date": session["date"],
                    "reference_low": m.get("reference_low"),
                    "reference_high": m.get("reference_high"),
                })
                break
    return [r for r in readings if r["value"] is not None]


def _months_between(date_str1: str, date_str2: str) -> float:
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        d1 = datetime.strptime(date_str1[:19], fmt)
        d2 = datetime.strptime(date_str2[:19], fmt)
        return max((d2 - d1).days / 30.44, 0.1)
    except Exception:
        return 1.0


def _project_threshold(current: float, monthly_rate: float, ref_low: float | None, ref_high: float | None) -> str | None:
    if monthly_rate == 0:
        return None
    if monthly_rate > 0 and ref_high is not None and current < ref_high:
        months = (ref_high - current) / monthly_rate
        return f"~{months:.0f} months until above normal range" if months < 24 else None
    if monthly_rate < 0 and ref_low is not None and current > ref_low:
        months = (current - ref_low) / abs(monthly_rate)
        return f"~{months:.0f} months until below normal range" if months < 24 else None
    return None


def analyze_trends(state: ReportState) -> dict:
    patient_id = state.get("patient_id", "")
    session_id = state.get("session_id")
    current_markers = state.get("final_markers", [])

    if not patient_id:
        log.info("[Node 6] No patient_id — skipping longitudinal analysis")
        return {"trend_results": []}

    history = get_patient_history(patient_id, exclude_session=session_id)
    if not history:
        log.info("[Node 6] No prior reports for patient_id=%s — first upload", patient_id)
        return {"trend_results": []}

    log.info("[Node 6] Trend analysis | patient=%s | prior_sessions=%d | current_markers=%d",
             patient_id, len(history), len(current_markers))

    results = []
    for marker in current_markers:
        name = marker.get("name", "")
        current_val = marker.get("value")
        if current_val is None:
            continue

        prior_readings = _find_marker_in_history(name, history)
        if not prior_readings:
            continue

        prev = prior_readings[-1]
        prev_val = prev["value"]
        ref_low = marker.get("reference_low") or prev.get("reference_low")
        ref_high = marker.get("reference_high") or prev.get("reference_high")
        unit = marker.get("unit", "")

        change_pct = round(((current_val - prev_val) / prev_val) * 100, 1) if prev_val else 0.0
        months = _months_between(prev["date"], history[-1]["date"]) if len(history) > 1 else 1.0
        monthly_rate = round((current_val - prev_val) / months, 3) if months > 0 else None
        months_to_threshold = _project_threshold(current_val, monthly_rate or 0, ref_low, ref_high)

        history_for_prompt = [{"value": r["value"], "date": r["date"][:10]} for r in prior_readings]
        history_for_prompt.append({"value": current_val, "date": "today"})

        trajectory = "STABLE"
        interpretation = f"{name} changed {change_pct:+.1f}% from {prev_val} to {current_val} {unit}."

        try:
            response = llm.invoke([HumanMessage(content=INTERPRET_PROMPT.format(
                name=name, unit=unit,
                history=history_for_prompt,
                current=current_val,
                ref_low=ref_low, ref_high=ref_high,
                change_percent=change_pct,
                monthly_rate=f"{monthly_rate:+.3f} {unit}/month" if monthly_rate is not None else "N/A",
                months_to_threshold=months_to_threshold or "Not projected within 2 years",
            ))])
            parsed = parse_llm_json(response.content)
            raw_trajectory = parsed.get("trajectory", "STABLE")
            trajectory = raw_trajectory if raw_trajectory in VALID_TRAJECTORIES else "STABLE"
            if raw_trajectory not in VALID_TRAJECTORIES:
                log.warning("[Node 6] Unexpected trajectory '%s' for %s — defaulting STABLE", raw_trajectory, name)
            interpretation = parsed.get("interpretation", interpretation)
        except Exception as e:
            log.warning("[Node 6] Interpretation LLM failed | marker=%s | error=%s", name, e)

        trend = {
            "marker_name": name,
            "unit": unit,
            "previous_value": prev_val,
            "current_value": current_val,
            "change_percent": change_pct,
            "monthly_rate": monthly_rate,
            "trajectory": trajectory,
            "months_to_threshold": months_to_threshold,
            "interpretation": interpretation,
        }
        results.append(trend)
        log.debug("[Node 6] Trend | %s | %s→%s %s | change=%+.1f%% | trajectory=%s",
                  name, prev_val, current_val, unit, change_pct, trajectory)

    log.info("[Node 6] Trend analysis complete | patient=%s | trends=%d", patient_id, len(results))
    return {"trend_results": results}
