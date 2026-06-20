from typing import Annotated, TypedDict
import operator


class ReportState(TypedDict):
    session_id: str
    patient_id: str                             # groups reports for longitudinal analysis
    file_path: str
    file_type: str
    raw_text: str                               # Node 1 output
    detected_panels: list[str]                  # Node 2 output
    extracted_markers: Annotated[list[dict], operator.add]  # merged from parallel extractors
    final_markers: list[dict]                   # Node 4 output — deduplicated, clean
    anomaly_tags: list[dict]                    # Node 4 output
    extraction_passed: bool                     # Critic output
    extraction_issues: list[str]                # Critic output
    extraction_retry_count: int
    pattern_matches: list[dict]                 # Node 5 output
    trend_results: list[dict]                   # Node 6 output
    doctor_briefing: dict                       # Node 7 output


class ChatState(TypedDict):
    session_id: str
    extracted_markers: list[dict]               # loaded from DB
    anomaly_tags: list[dict]                    # loaded from DB
    chat_history: Annotated[list[dict], operator.add]   # accumulates across turns
    question: str                               # current user message
    query_intent: str                           # Node 5 output
    context: dict                               # Node 6 output
    response: str                               # Node 7 output
