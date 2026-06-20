import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.models.schemas import AnomalyTag
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

SEVERITY_PROMPT = """You are a clinical pathologist reviewing abnormal lab results.

For each marker below, assign a clinical severity:
- CRITICAL_HIGH or CRITICAL_LOW: requires immediate medical attention (e.g. risk of cardiac arrhythmia, seizure, acute organ failure)
- HIGH or LOW: clinically abnormal and worth addressing, but not an immediate emergency

Base this on your medical knowledge of each specific marker — not on deviation percentage alone.
A 15% elevation in serum potassium can be more dangerous than a 300% elevation in bilirubin.

Abnormal markers:
{markers}

Respond with a JSON array only. No explanation. Example:
[
  {{"name": "Sodium", "status": "CRITICAL_HIGH"}},
  {{"name": "Glucose Fasting", "status": "HIGH"}}
]"""


def tag_anomalies(state: ReportState) -> dict:
    raw_markers = state.get("extracted_markers", [])

    seen = {}
    for m in raw_markers:
        seen[(m.get("name"), m.get("panel"))] = m
    markers = list(seen.values())

    if len(raw_markers) != len(markers):
        log.debug("[Node 4] Deduplicated markers | before=%d | after=%d", len(raw_markers), len(markers))

    log.info("[Node 4] Tagging anomalies | session=%s | markers=%d", state.get("session_id"), len(markers))

    out_of_range = []
    for marker in markers:
        value = marker.get("value")
        ref_low = marker.get("reference_low")
        ref_high = marker.get("reference_high")
        if value is None:
            continue

        if ref_high is not None and value > ref_high:
            deviation = round(((value - ref_high) / ref_high) * 100, 1)
            out_of_range.append({
                "name": marker["name"],
                "value": value,
                "unit": marker.get("unit") or "",
                "direction": "HIGH",
                "deviation_percent": deviation,
                "ref_low": ref_low,
                "ref_high": ref_high,
            })
        elif ref_low is not None and value < ref_low:
            deviation = round(((ref_low - value) / ref_low) * 100, 1)
            out_of_range.append({
                "name": marker["name"],
                "value": value,
                "unit": marker.get("unit") or "",
                "direction": "LOW",
                "deviation_percent": deviation,
                "ref_low": ref_low,
                "ref_high": ref_high,
            })

    if not out_of_range:
        log.info("[Node 4] No anomalies detected")
        return {"anomaly_tags": [], "final_markers": markers}

    descriptions = [
        f"- {m['name']}: {m['value']} {m['unit']} "
        f"(ref {m['ref_low']}–{m['ref_high']}, {m['deviation_percent']:+.1f}% from upper limit)"
        for m in out_of_range
    ]

    severity_map = {}
    try:
        response = llm.invoke([HumanMessage(content=SEVERITY_PROMPT.format(
            markers="\n".join(descriptions)
        ))])
        assessments = parse_llm_json(response.content)
        for a in assessments:
            severity_map[a["name"]] = a["status"]
        log.debug("[Node 4] LLM severity assessments: %s", severity_map)
    except Exception as e:
        log.warning("[Node 4] Severity LLM failed — using direction only | error=%s", e)
        for m in out_of_range:
            severity_map[m["name"]] = m["direction"]

    valid_statuses = {"HIGH", "LOW", "CRITICAL_HIGH", "CRITICAL_LOW"}
    tags = []
    for m in out_of_range:
        status = severity_map.get(m["name"], m["direction"])
        if status not in valid_statuses:
            status = m["direction"]
        tags.append(AnomalyTag(
            marker_name=m["name"],
            value=m["value"],
            unit=m["unit"],
            status=status,
            deviation_percent=m["deviation_percent"],
        ).model_dump())

    log.info("[Node 4] Tagging complete | session=%s | anomalies=%d", state.get("session_id"), len(tags))
    for tag in tags:
        log.debug("[Node 4] Anomaly | %s = %s %s | status=%s | deviation=%s%%",
                  tag["marker_name"], tag["value"], tag["unit"], tag["status"], tag.get("deviation_percent"))
    return {"anomaly_tags": tags, "final_markers": markers}
