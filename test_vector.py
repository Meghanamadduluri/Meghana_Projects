from app.db.vector_store import VectorStore

vs = VectorStore()

docs = [
    "FastAPI is a modern Python web framework",
    "FastAPI uses Pydantic for validation",
    "Vector databases store embeddings",
]

vs.add_document("id_1", docs[0])

results = vs.search("What does FastAPI use for validation?")

print(results)