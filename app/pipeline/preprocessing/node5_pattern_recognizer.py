import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from app.pipeline.state import ReportState
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

SYSTEM_PROMPT = """You are a clinical pathologist analysing a patient's lab results.

Your job is to identify any clinically significant patterns, syndromes, or correlations present
in the data — using your own medical knowledge. You are NOT given a list of patterns to check against.
You reason from the data itself.

You have one tool available:
- report_pattern: call this once for each distinct clinical pattern you identify

Rules:
- Only report patterns supported by the actual marker values in the data
- A pattern needs at least 2 markers to be meaningful — single-marker findings are just anomalies, not patterns
- Exception: a single highly specific marker (e.g. TSH alone strongly suggests thyroid disorder) may qualify
- Confidence: HIGH = clear multi-marker evidence, MODERATE = partial evidence worth discussing
- Do not report the same condition twice under different names
- When done, stop calling tools and output only: DONE"""

MARKERS_MESSAGE = """Here are the patient's extracted lab markers:

{markers}

Analyse these and report any clinical patterns you identify using the report_pattern tool.
Call the tool once per pattern. When finished, output: DONE"""


def make_report_tool(matches: list):
    @tool
    def report_pattern(
        name: str,
        confidence: str,
        evidence: list[str],
        implication: str,
        differentials: list[str],
    ) -> str:
        """Report a clinical pattern identified in the lab results.

        Args:
            name: Name of the clinical pattern or syndrome (e.g. "Type 2 Diabetes", "Iron Deficiency Anemia")
            confidence: "HIGH" or "MODERATE"
            evidence: List of specific marker values supporting this pattern (reference actual numbers)
            implication: One plain-English sentence explaining what this means for the patient
            differentials: Other conditions that could explain the same findings
        """
        matches.append({
            "name": name,
            "confidence": confidence,
            "evidence": evidence,
            "implication": implication,
            "differentials": differentials,
        })
        log.debug("[Node 5] Pattern reported | name=%s | confidence=%s", name, confidence)
        return f"Pattern '{name}' recorded."

    return report_pattern


def recognize_patterns(state: ReportState) -> dict:
    markers = state.get("final_markers", [])
    session_id = state.get("session_id")
    log.info("[Node 5] Agentic pattern recognition | session=%s | markers=%d", session_id, len(markers))

    if not markers:
        log.info("[Node 5] No markers — skipping")
        return {"pattern_matches": []}

    matches = []
    report_tool = make_report_tool(matches)
    llm_with_tools = llm.bind_tools([report_tool])

    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=MARKERS_MESSAGE.format(markers=json.dumps(markers, indent=2))),
    ]

    iterations = 0
    max_iterations = 8

    while iterations < max_iterations:
        iterations += 1
        log.debug("[Node 5] ReAct iteration %d", iterations)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            log.debug("[Node 5] Agent finished | final_text=%.80s", response.content)
            break

        for call in response.tool_calls:
            log.debug("[Node 5] Tool call | tool=%s | name=%s", call["name"], call["args"].get("name"))
            result = report_tool.invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    log.info("[Node 5] Pattern recognition complete | session=%s | patterns=%d | names=%s",
             session_id, len(matches), [m["name"] for m in matches])
    return {"pattern_matches": matches}
