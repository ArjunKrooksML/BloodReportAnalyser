import logging
from app.pipeline.state import ReportState
from app.models.schemas import AnomalyTag

log = logging.getLogger(__name__)


def tag_anomalies(state: ReportState) -> dict:
    raw_markers = state.get("extracted_markers", [])

    # deduplicate by (name, panel) keeping the last occurrence (most recent retry wins)
    seen = {}
    for m in raw_markers:
        seen[(m.get("name"), m.get("panel"))] = m
    markers = list(seen.values())

    if len(raw_markers) != len(markers):
        log.debug("[Node 4] Deduplicated markers | before=%d | after=%d", len(raw_markers), len(markers))

    log.info("[Node 4] Tagging anomalies | session=%s | markers=%d", state.get("session_id"), len(markers))
    tags = []

    for marker in markers:
        value = marker.get("value")
        ref_low = marker.get("reference_low")
        ref_high = marker.get("reference_high")

        if value is None:
            continue

        status = None
        deviation = None

        if ref_high is not None and value > ref_high:
            deviation = round(((value - ref_high) / ref_high) * 100, 1)
            status = "CRITICAL_HIGH" if deviation > 50 else "HIGH"
        elif ref_low is not None and value < ref_low:
            deviation = round(((ref_low - value) / ref_low) * 100, 1)
            status = "CRITICAL_LOW" if deviation > 50 else "LOW"

        if status:
            tags.append(AnomalyTag(
                marker_name=marker["name"],
                value=value,
                unit=marker.get("unit", ""),
                status=status,
                deviation_percent=deviation
            ).model_dump())

    log.info("[Node 4] Tagging complete | session=%s | anomalies=%d", state.get("session_id"), len(tags))
    for tag in tags:
        log.debug("[Node 4] Anomaly | %s = %s %s | status=%s | deviation=%s%%",
                  tag["marker_name"], tag["value"], tag["unit"], tag["status"], tag.get("deviation_percent"))
    return {"anomaly_tags": tags}
