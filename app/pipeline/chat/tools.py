from langchain_core.tools import tool
from app.pipeline.state import ChatState


def make_tools(state: ChatState):

    @tool
    def lookup_marker(name: str) -> dict:
        """Fetch a specific marker by name from the patient's report."""
        markers = state.get("extracted_markers", [])
        name_lower = name.lower()
        matches = [m for m in markers if name_lower in m["name"].lower()]
        return matches[0] if matches else {"error": f"Marker '{name}' not found in report."}

    @tool
    def get_all_markers() -> list[dict]:
        """Return all extracted markers from the patient's report."""
        return state.get("extracted_markers", [])

    @tool
    def get_all_anomalies() -> list[dict]:
        """Return all markers flagged as abnormal in the patient's report."""
        return state.get("anomaly_tags", [])

    @tool
    def check_range(name: str) -> str:
        """Check whether a specific marker is within, above, or below its reference range."""
        markers = state.get("extracted_markers", [])
        name_lower = name.lower()
        marker = next((m for m in markers if name_lower in m["name"].lower()), None)

        if not marker:
            return f"Marker '{name}' not found."

        value = marker.get("value")
        low = marker.get("reference_low")
        high = marker.get("reference_high")

        if low is None and high is None:
            return f"{marker['name']}: {value} {marker.get('unit', '')} — no reference range available."
        if high is not None and value > high:
            return f"{marker['name']}: {value} {marker.get('unit', '')} is ABOVE the normal range ({low}–{high})."
        if low is not None and value < low:
            return f"{marker['name']}: {value} {marker.get('unit', '')} is BELOW the normal range ({low}–{high})."
        return f"{marker['name']}: {value} {marker.get('unit', '')} is within the normal range ({low}–{high})."

    @tool
    def get_panel_markers(panel: str) -> list[dict]:
        """Return all markers belonging to a specific test panel (e.g. CBC, LFT)."""
        markers = state.get("extracted_markers", [])
        return [m for m in markers if panel.lower() in m.get("panel", "").lower()]

    return [lookup_marker, get_all_markers, get_all_anomalies, check_range, get_panel_markers]
