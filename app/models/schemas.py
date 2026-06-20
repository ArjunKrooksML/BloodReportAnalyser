from pydantic import BaseModel
from typing import Optional
from enum import Enum


class QueryIntent(str, Enum):
    RANGE_CHECK = "range_check"
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"
    SUMMARY = "summary"


class Marker(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = ""
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    panel: str


class AnomalyTag(BaseModel):
    marker_name: str
    value: float
    unit: str
    status: str
    deviation_percent: Optional[float]


class PatternMatch(BaseModel):
    name: str
    confidence: str                   # "HIGH", "MODERATE", "LOW"
    evidence: list[str]               # specific marker values that support this pattern
    implication: str                  # plain-English clinical meaning
    differentials: list[str]          # other conditions to rule out


class MarkerTrend(BaseModel):
    marker_name: str
    unit: str
    previous_value: float
    current_value: float
    change_percent: float
    monthly_rate: Optional[float]     # change per month
    trajectory: str                   # "IMPROVING", "STABLE", "WORSENING", "CRITICAL_TREND"
    months_to_threshold: Optional[str]
    interpretation: str               # LLM-generated clinical narrative


class DoctorBriefing(BaseModel):
    urgency_level: str                # "CRITICAL", "HIGH", "MODERATE", "LOW"
    summary: str
    critical_now: list[str]
    watch_list: list[str]
    positive_findings: list[str]
    questions_for_doctor: list[str]   # value-specific, not generic
    lifestyle_notes: list[str]


class UploadResponse(BaseModel):
    session_id: str
    patient_id: Optional[str] = None
    detected_panels: list[str]
    extracted_markers: list[Marker]
    anomaly_tags: list[AnomalyTag]
    pattern_matches: list[PatternMatch] = []
    trend_results: list[MarkerTrend] = []
    doctor_briefing: Optional[DoctorBriefing] = None


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    response: str
    intent: QueryIntent
    relevant_markers: list[Marker]
