import uuid
import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from langgraph.types import Command

from app.pipeline.graphs.preprocessing_graph import preprocessing_graph
from app.pipeline.graphs.chat_graph import chat_graph
from app.models.schemas import UploadResponse, ChatRequest, ChatResponse
from app.db.store import save_session, load_session, load_chat_history, save_chat_turn

log = logging.getLogger(__name__)
router = APIRouter()

UPLOADS_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@router.post("/upload", response_model=UploadResponse)
async def upload_report(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    log.info("Upload received | filename=%s | type=%s", file.filename, suffix)

    if suffix not in ALLOWED_EXTENSIONS:
        log.warning("Rejected file type: %s", suffix)
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    session_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{session_id}{suffix}"

    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    log.debug("File saved | session=%s | path=%s", session_id, file_path)

    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id": session_id,
        "file_path": str(file_path),
        "file_type": suffix.lstrip("."),
        "extraction_retry_count": 0
    }

    log.info("Starting preprocessing graph | session=%s", session_id)
    result = await preprocessing_graph.ainvoke(initial_state, config=config)

    if isinstance(result, dict) and result.get("__interrupt__"):
        interrupt_data = result["__interrupt__"][0].value
        log.warning("Graph interrupted — needs clarification | session=%s | reason=%s", session_id, interrupt_data.get("reason"))
        return {
            "session_id": session_id,
            "needs_clarification": True,
            "clarification_prompt": interrupt_data.get("prompt"),
            "reason": interrupt_data.get("reason"),
            "detected_panels": [],
            "extracted_markers": [],
            "anomaly_tags": []
        }

    panels = result.get("detected_panels", [])
    markers = result.get("extracted_markers", [])
    anomalies = result.get("anomaly_tags", [])
    log.info("Preprocessing complete | session=%s | panels=%s | markers=%d | anomalies=%d",
             session_id, panels, len(markers), len(anomalies))

    save_session(session_id=session_id, detected_panels=panels, extracted_markers=markers, anomaly_tags=anomalies)
    log.debug("Session persisted to DB | session=%s", session_id)

    return UploadResponse(session_id=session_id, detected_panels=panels, extracted_markers=markers, anomaly_tags=anomalies)


@router.post("/clarify/{session_id}", response_model=UploadResponse)
async def clarify_report(session_id: str, clarification: str):
    log.info("Clarification received | session=%s | text=%.80s", session_id, clarification)
    config = {"configurable": {"thread_id": session_id}}

    log.debug("Resuming preprocessing graph | session=%s", session_id)
    result = await preprocessing_graph.ainvoke(Command(resume=clarification), config=config)

    panels = result.get("detected_panels", [])
    markers = result.get("extracted_markers", [])
    anomalies = result.get("anomaly_tags", [])
    log.info("Clarification reprocessing complete | session=%s | panels=%s | markers=%d | anomalies=%d",
             session_id, panels, len(markers), len(anomalies))

    save_session(session_id=session_id, detected_panels=panels, extracted_markers=markers, anomaly_tags=anomalies)
    return UploadResponse(session_id=session_id, detected_panels=panels, extracted_markers=markers, anomaly_tags=anomalies)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    log.info("Chat request | session=%s | question=%.80s", request.session_id, request.question)

    session = load_session(request.session_id)
    if not session:
        log.warning("Session not found | session=%s", request.session_id)
        raise HTTPException(status_code=404, detail="Session not found. Please upload a report first.")

    chat_history = load_chat_history(request.session_id)
    log.debug("Loaded session | markers=%d | anomalies=%d | history_turns=%d",
              len(session["extracted_markers"]), len(session["anomaly_tags"]), len(chat_history))

    config = {"configurable": {"thread_id": f"chat-{request.session_id}"}}
    initial_state = {
        "session_id": request.session_id,
        "extracted_markers": session["extracted_markers"],
        "anomaly_tags": session["anomaly_tags"],
        "chat_history": chat_history,
        "question": request.question,
        "query_intent": "",
        "context": {},
        "response": ""
    }

    log.debug("Starting chat agent | session=%s", request.session_id)
    result = await chat_graph.ainvoke(initial_state, config=config)

    save_chat_turn(request.session_id, request.question, result["response"])
    log.info("Chat turn complete | session=%s | response_chars=%d", request.session_id, len(result["response"]))

    return ChatResponse(
        response=result["response"],
        intent=result.get("query_intent") or "summary",
        relevant_markers=result.get("context", {}).get("markers", [])
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    log.debug("Session fetch | session=%s", session_id)
    session = load_session(session_id)
    if not session:
        log.warning("Session not found | session=%s", session_id)
        raise HTTPException(status_code=404, detail="Session not found.")
    return session
