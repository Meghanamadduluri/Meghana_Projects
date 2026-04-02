import fitz
import os
from app.services.chunking import chunk_text

def extract_text_from_pdf(pdf_path: str):

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:
        page_text = page.get_text()
        full_text += page_text

    return full_text

def process_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"chunks created for {pdf_path}: {len(chunks)}")
    paper_name = os.path.basename(pdf_path)
    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{paper_name}_{i}"
        documents.append({
            "doc_id": doc_id,
            "text": chunk,
            "paper": paper_name,
            "chunk_id": i
        })
    return documents