import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=512)

CRITIC_PROMPT = """You are a data quality checker for extracted lab markers. Your job is extremely narrow.

FLAG only these two things:
1. The value field contains non-numeric garbage that cannot be a real measurement (e.g. "2a8", "N/A", "---", "..")
2. The value is so extreme it is biologically impossible for a human to be alive (e.g. Hemoglobin 850 g/dL, Glucose 9000 mg/dL, WBC 0.000001 K/μL)

NEVER flag any of these — they are all valid:
- A value that is higher than reference_high or lower than reference_low → this is just an abnormal result, not an error
- A value slightly above or below the reference range by any amount → valid clinical data
- A ratio or index marker (e.g. LDL/HDL Ratio, Cholesterol/HDL) with no unit → ratios are dimensionless
- reference_low or reference_high being null → one-sided ranges are normal (eGFR, ESR, etc.)
- Capitalisation or abbreviation differences in marker names (HbA1c vs HbA1C, eGFR vs EGFR) → not OCR errors
- Any marker name that looks like a real medical test, even if uncommon

Examples that PASS (return passed: true):
- RDW CV: 14.4%  ref 11.0–14.0%  ← slightly above range, valid
- ESR: 46 mm/h  ref high 10 mm/h  ← very elevated but biologically possible
- Phosphorus: 5.3 mg/dL  ref high 4.5  ← valid elevated phosphorus
- Cholesterol/HDL Ratio: 4.2  unit: null  ← dimensionless ratio, correct
- HbA1C: 7.2%  ← capitalisation variant, fine

Examples that FAIL (return passed: false):
- Hemoglobin: 850 g/dL  ← no living human has this
- Glucose: "2a8 mg/dL"  ← OCR garbage, not a parseable number

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
