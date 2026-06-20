import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=256)

CLASSIFY_PROMPT = """You are analyzing raw text from a medical lab report.
Identify all distinct test panels or test groups present in the text.

Use common medical shorthand where applicable (e.g. CBC, LFT, KFT, HbA1c, Lipid Profile, Thyroid, Iron Studies, Vitamin D).
If a panel does not match a known shorthand, name it descriptively (e.g. "Urine Routine", "Hormone Panel").

Respond with JSON only:
{{
  "detected_panels": ["panel1", "panel2", ...]
}}

Raw text:
{text}"""


def classify_report(state: ReportState) -> dict:
    log.info("[Node 2] Classifying report | session=%s", state["session_id"])

    response = llm.invoke([HumanMessage(content=CLASSIFY_PROMPT.format(text=state["raw_text"]))])

    try:
        result = parse_llm_json(response.content)
        panels = result.get("detected_panels", ["UNKNOWN"])
    except Exception:
        log.warning("[Node 2] Failed to parse classifier response — defaulting to UNKNOWN | raw=%.200s", response.content)
        panels = ["UNKNOWN"]

    log.info("[Node 2] Detected panels | session=%s | panels=%s", state["session_id"], panels)
    return {"detected_panels": panels}
