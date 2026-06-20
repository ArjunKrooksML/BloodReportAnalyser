from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.pipeline.state import ReportState
from app.pipeline.preprocessing.node1_parser import parse_document
from app.pipeline.preprocessing.node1b_clarifier import check_clarity
from app.pipeline.preprocessing.node2_classifier import classify_report
from app.pipeline.preprocessing.node3_extractor import fan_out_extractions, extract_panel
from app.pipeline.preprocessing.node3_critic import critique_extraction, should_retry
from app.pipeline.preprocessing.node4_tagger import tag_anomalies
from app.pipeline.preprocessing.node5_pattern_recognizer import recognize_patterns
from app.pipeline.preprocessing.node6_trend_engine import analyze_trends
from app.pipeline.preprocessing.node7_briefing_generator import generate_briefing


def prepare_retry(state: ReportState) -> dict:
    return {}


builder = StateGraph(ReportState)

builder.add_node("parse_document",       parse_document)
builder.add_node("check_clarity",        check_clarity)
builder.add_node("classify_report",      classify_report)
builder.add_node("extract_panel",        extract_panel)
builder.add_node("critique_extraction",  critique_extraction)
builder.add_node("prepare_retry",        prepare_retry)
builder.add_node("tag_anomalies",        tag_anomalies)
builder.add_node("recognize_patterns",   recognize_patterns)
builder.add_node("analyze_trends",       analyze_trends)
builder.add_node("generate_briefing",    generate_briefing)

builder.set_entry_point("parse_document")
builder.add_edge("parse_document",      "check_clarity")
builder.add_edge("check_clarity",       "classify_report")

builder.add_conditional_edges("classify_report", fan_out_extractions, ["extract_panel"])
builder.add_edge("extract_panel", "critique_extraction")
builder.add_conditional_edges("critique_extraction", should_retry, {
    "prepare_retry": "prepare_retry",
    "tag_anomalies": "tag_anomalies"
})
builder.add_conditional_edges("prepare_retry", fan_out_extractions, ["extract_panel"])

# parallel fork: pattern recognition and trend analysis run simultaneously
builder.add_edge("tag_anomalies",       "recognize_patterns")
builder.add_edge("tag_anomalies",       "analyze_trends")

# both converge at briefing generator
builder.add_edge("recognize_patterns",  "generate_briefing")
builder.add_edge("analyze_trends",      "generate_briefing")
builder.add_edge("generate_briefing",   END)

checkpointer = MemorySaver()
preprocessing_graph = builder.compile(checkpointer=checkpointer)
