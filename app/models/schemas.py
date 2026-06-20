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
    status: str                       # "HIGH", "LOW", "CRITICAL_HIGH", "CRITICAL_LOW"
    deviation_percent: Optional[float]


class UploadResponse(BaseModel):
    session_id: str
    detected_panels: list[str]        # free-form, e.g. ["CBC", "Vitamin D", "Iron Studies"]
    extracted_markers: list[Marker]
    anomaly_tags: list[AnomalyTag]


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    response: str
    intent: QueryIntent
    relevant_markers: list[Marker]
