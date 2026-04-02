from app.services.pdf_processor import process_pdf
from app.db.vector_store import VectorStore
import os

vector_store = VectorStore()

papers_dir = "papers"

for filename in os.listdir(papers_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(papers_dir, filename)
        documents = process_pdf(pdf_path)
        for doc in documents:
            vector_store.add_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                paper=doc["paper"],
                chunk_id=doc["chunk_id"]
            )
print("Finished ingesting PDFs")
print(f"Total chunks:", vector_store.collection.count())