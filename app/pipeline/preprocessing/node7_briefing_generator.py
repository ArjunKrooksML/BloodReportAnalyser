import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.utils.json_utils import parse_llm_json

log = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o", max_tokens=2048)

BRIEFING_PROMPT = """You are preparing a medical briefing for a patient to bring to their next doctor's appointment.

REPORT DATA:
Markers extracted: {markers}
Anomalies flagged: {anomalies}
Clinical patterns detected: {patterns}
Longitudinal trends: {trends}

Generate a structured briefing following these STRICT rules:

URGENCY LEVELS:
- CRITICAL: values in critical range (>50% outside reference) or critical trend detected
- HIGH: significant anomalies or worsening trends
- MODERATE: mild anomalies, patterns detected, or early trends
- LOW: all within range, no patterns

QUESTIONS FOR DOCTOR — must be VALUE-SPECIFIC:
- Bad: "Ask about your glucose levels"
- Good: "My Glucose Postprandial is 228 mg/dL — 62.9% above the upper limit of 140. Should I start medication now or try lifestyle changes first?"
- If trends exist: "My [marker] has changed from [prev] to [current] in [time]. At this rate, when should I be concerned?"

POSITIVE FINDINGS — only genuine ones, never fabricated:
- Only list markers that are genuinely within range
- If everything is abnormal, say so honestly

LIFESTYLE NOTES — must be tied to specific findings:
- Bad: "Exercise more and eat healthily"
- Good: "A Glucose Postprandial of 228 suggests post-meal spikes — a 15-minute walk after meals can reduce blood sugar by 20-30 mg/dL"

Respond with JSON only:
{{
  "urgency_level": "CRITICAL|HIGH|MODERATE|LOW",
  "summary": "2-3 sentences in plain English, no jargon",
  "critical_now": ["list of things needing immediate attention"],
  "watch_list": ["things to monitor at next appointment"],
  "positive_findings": ["what looks good"],
  "questions_for_doctor": ["specific value-referenced questions"],
  "lifestyle_notes": ["actionable suggestions tied to specific findings"]
}}"""


def generate_briefing(state: ReportState) -> dict:
    markers = state.get("final_markers", [])
    anomalies = state.get("anomaly_tags", [])
    patterns = state.get("pattern_matches", [])
    trends = state.get("trend_results", [])
    session_id = state.get("session_id")

    log.info("[Node 7] Generating doctor briefing | session=%s | markers=%d | anomalies=%d | patterns=%d | trends=%d",
             session_id, len(markers), len(anomalies), len(patterns), len(trends))

    prompt = BRIEFING_PROMPT.format(
        markers=json.dumps(markers, indent=2),
        anomalies=json.dumps(anomalies, indent=2),
        patterns=json.dumps(patterns, indent=2) if patterns else "None detected",
        trends=json.dumps(trends, indent=2) if trends else "No prior reports available for trend analysis",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        briefing = parse_llm_json(response.content)
        log.info("[Node 7] Briefing generated | session=%s | urgency=%s | questions=%d",
                 session_id, briefing.get("urgency_level"), len(briefing.get("questions_for_doctor", [])))
        return {"doctor_briefing": briefing}
    except Exception as e:
        log.warning("[Node 7] Failed to parse briefing | error=%s | raw=%.200s", e, response.content)
        return {"doctor_briefing": {}}
