import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.pipeline.state import ReportState
from app.utils.pdf_utils import (
    extract_pdf_text, is_scanned_pdf,
    pdf_to_base64_images, image_to_base64
)

log = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

VISION_PROMPT = "Transcribe all visible text from this image exactly as written. Include every number, label, unit, and measurement you can see. Do not summarize or omit anything."


def parse_document(state: ReportState) -> dict:
    file_path = state["file_path"]
    suffix = Path(file_path).suffix.lower()
    log.info("[Node 1] Parsing document | session=%s | file=%s", state["session_id"], file_path)

    if suffix == ".pdf":
        if not is_scanned_pdf(file_path):
            log.info("[Node 1] Digital PDF detected — extracting text layer directly")
            raw_text = extract_pdf_text(file_path)
            log.debug("[Node 1] Text layer extracted | chars=%d", len(raw_text))
            log.debug("[Node 1] Raw text preview | %s", raw_text[:300])
            return {"raw_text": raw_text, "file_type": "pdf"}

        log.info("[Node 1] Scanned PDF detected — falling back to vision")
        images = pdf_to_base64_images(file_path)
    else:
        log.info("[Node 1] Image file — using vision")
        images = image_to_base64(file_path)

    log.debug("[Node 1] Sending %d page(s) to vision model", len(images))
    content = [{"type": "text", "text": VISION_PROMPT}]
    for b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
        })

    response = llm.invoke([HumanMessage(content=content)])
    log.debug("[Node 1] Vision extraction complete | chars=%d", len(response.content))
    log.debug("[Node 1] Raw text preview | %s", response.content[:300])

    return {"raw_text": response.content, "file_type": suffix.lstrip(".")}
