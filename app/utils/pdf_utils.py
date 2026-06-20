import pypdfium2 as pdfium
import base64
from pathlib import Path
from PIL import Image
import io

MIN_TEXT_CHARS = 100  # below this we treat the PDF as a scanned image


def extract_pdf_text(file_path: str) -> str:
    """Extract text directly from a PDF's text layer. Returns empty string if none."""
    pdf = pdfium.PdfDocument(file_path)
    pages = []
    for page in pdf:
        textpage = page.get_textpage()
        pages.append(textpage.get_text_range())
    return "\n\n".join(pages)


def is_scanned_pdf(file_path: str) -> bool:
    return len(extract_pdf_text(file_path).strip()) < MIN_TEXT_CHARS


def pdf_to_base64_images(file_path: str) -> list[str]:
    pdf = pdfium.PdfDocument(file_path)
    images = []
    for page in pdf:
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        images.append(_pil_to_base64(pil_image))
    return images


def image_to_base64(file_path: str) -> list[str]:
    with Image.open(file_path) as img:
        return [_pil_to_base64(img)]


def _pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
