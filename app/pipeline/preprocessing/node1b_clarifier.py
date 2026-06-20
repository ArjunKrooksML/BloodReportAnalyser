import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from app.pipeline.state import ReportState

log = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=256)

CLARITY_PROMPT = """You are reviewing raw text extracted from a medical lab report.
Assess whether the text is clear enough to identify test names, values, and units.

Respond with JSON only:
{{
  "is_clear": true or false,
  "reason": "one sentence explanation if unclear, else null"
}}

Raw text:
{text}"""


def check_clarity(state: ReportState) -> dict:
    log.info("[Node 1b] Checking clarity | session=%s", state["session_id"])

    response = llm.invoke([HumanMessage(content=CLARITY_PROMPT.format(text=state["raw_text"][:2000]))])

    try:
        result = json.loads(response.content)
    except Exception:
        log.warning("[Node 1b] Could not parse clarity response — assuming clear")
        return {}

    is_clear = result.get("is_clear", True)
    reason = result.get("reason")
    log.debug("[Node 1b] Clarity result | is_clear=%s | reason=%s", is_clear, reason)

    if not is_clear:
        log.warning("[Node 1b] Report unclear — interrupting for user clarification | reason=%s", reason)
        user_input = interrupt({
            "message": "The uploaded report is difficult to read.",
            "reason": reason,
            "prompt": "Please describe what type of report this is and what tests it contains, so we can extract the data correctly."
        })
        log.info("[Node 1b] Clarification received — appending to raw text")
        return {"raw_text": state["raw_text"] + f"\n\nUser clarification: {user_input}"}

    log.info("[Node 1b] Report is clear — proceeding")
    return {}
