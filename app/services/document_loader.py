from pypdf import PdfReader
from app.services.chunking import chunk_text

def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_and_chunk_pdf(file_path: str):
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    return chunks