import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512)

CRITIC_PROMPT = """You are reviewing extracted markers from a medical lab report.
Check ONLY for these structural issues — do NOT flag clinically abnormal values as errors:
- Missing units on any marker
- Values that are physically impossible (e.g. hemoglobin of 500 g/dL, glucose of 5000 mg/dL)
- Markers where the value is clearly an OCR error (e.g. letters mixed into a number like "2a8")
- Obvious OCR errors in marker names (e.g. "Glucos3" instead of "Glucose")

Do NOT flag a value just because it is outside the reference range — abnormal lab results are valid data.

Respond with JSON only:
{{
  "passed": true or false,
  "issues": ["issue1", "issue2"] or []
}}

Extracted markers:
{markers}"""

MAX_RETRIES = 2


def critique_extraction(state: ReportState) -> dict:
    markers = state.get("extracted_markers", [])
    retry_count = state.get("extraction_retry_count", 0)
    log.info("[Critic] Reviewing extraction | session=%s | markers=%d | attempt=%d",
             state.get("session_id"), len(markers), retry_count + 1)

    response = llm.invoke([HumanMessage(content=CRITIC_PROMPT.format(markers=json.dumps(markers, indent=2)))])

    try:
        result = parse_llm_json(response.content)
    except Exception:
        log.warning("[Critic] Could not parse critic response — assuming passed | raw=%.200s", response.content)
        return {"extraction_passed": True}

    passed = result.get("passed", True)
    issues = result.get("issues", [])

    if passed:
        log.info("[Critic] Extraction passed | session=%s", state.get("session_id"))
    else:
        log.warning("[Critic] Extraction failed | session=%s | issues=%s", state.get("session_id"), issues)

    return {
        "extraction_passed": passed,
        "extraction_issues": issues,
        "extraction_retry_count": retry_count + (0 if passed else 1)
    }


def should_retry(state: ReportState) -> str:
    if state.get("extraction_passed", True):
        log.debug("[Critic] Decision: proceed to tagging | session=%s", state.get("session_id"))
        return "tag_anomalies"
    if state.get("extraction_retry_count", 0) >= MAX_RETRIES:
        log.warning("[Critic] Max retries reached — proceeding despite issues | session=%s", state.get("session_id"))
        return "tag_anomalies"
    log.info("[Critic] Decision: retry extraction | session=%s | retry=%d", state.get("session_id"), state.get("extraction_retry_count"))
    return "prepare_retry"
