from typing import Annotated, TypedDict
import operator


class ReportState(TypedDict):
    session_id: str
    file_path: str
    file_type: str                              # "pdf" or "image"
    raw_text: str                               # Node 1 output
    detected_panels: list[str]                  # Node 2 output e.g. ["CBC", "KFT"]
    extracted_markers: Annotated[list[dict], operator.add]  # Node 3 output — merged from parallel extractors
    anomaly_tags: list[dict]                    # Node 4 output
    extraction_passed: bool                     # Node 3 Critic output
    extraction_issues: list[str]                # Node 3 Critic output
    extraction_retry_count: int                 # tracks reflection loop iterations


class ChatState(TypedDict):
    session_id: str
    extracted_markers: list[dict]               # loaded from DB
    anomaly_tags: list[dict]                    # loaded from DB
    chat_history: Annotated[list[dict], operator.add]   # accumulates across turns
    question: str                               # current user message
    query_intent: str                           # Node 5 output
    context: dict                               # Node 6 output
    response: str                               # Node 7 output
