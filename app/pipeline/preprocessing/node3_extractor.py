import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import Send
from app.pipeline.state import ReportState
from app.models.schemas import Marker
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

EXTRACT_PROMPT = """Extract all test markers for the "{panel}" panel from the following lab report text.

For each marker return:
- name: test name as written
- value: numeric result (float)
- unit: measurement unit
- reference_low: lower bound of reference range (float or null)
- reference_high: upper bound of reference range (float or null)
- panel: "{panel}"

Respond with JSON only:
{{
  "markers": [
    {{"name": "...", "value": 0.0, "unit": "...", "reference_low": 0.0, "reference_high": 0.0, "panel": "{panel}"}}
  ]
}}

Lab report text:
{text}"""


def fan_out_extractions(state: ReportState) -> list[Send]:
    panels = state["detected_panels"]
    log.info("[Node 3] Fanning out extraction | session=%s | panels=%s", state.get("session_id"), panels)
    return [
        Send("extract_panel", {"panel": panel, "raw_text": state["raw_text"], "session_id": state.get("session_id")})
        for panel in panels
    ]


def extract_panel(state: dict) -> dict:
    panel = state["panel"]
    log.info("[Node 3] Extracting panel | session=%s | panel=%s", state.get("session_id"), panel)

    response = llm.invoke([HumanMessage(content=EXTRACT_PROMPT.format(panel=panel, text=state["raw_text"]))])

    try:
        result = parse_llm_json(response.content)
        all_markers = [Marker(**m) for m in result.get("markers", [])]
        markers = [m.model_dump() for m in all_markers if m.value is not None]
        skipped = len(all_markers) - len(markers)
        if skipped:
            log.debug("[Node 3] Skipped %d marker(s) with null value | panel=%s", skipped, panel)
        log.debug("[Node 3] Extracted markers | panel=%s | count=%d", panel, len(markers))
    except Exception as e:
        log.warning("[Node 3] Extraction failed for panel=%s | error=%s | raw=%.200s", panel, e, response.content)
        markers = []

    return {"extracted_markers": markers}
