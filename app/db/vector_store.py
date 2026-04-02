from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from numpy import argsort
from rank_bm25 import BM25Okapi

class VectorStore:

# Initialize the vector store with a sentence transformer model and ChromaDB client
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client(
            Settings(
                persist_directory="./chroma_db",
                is_persistent=True
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="documents"
        )
        self.corpus = []
        self.tokenized_corpus = []
        self.metadata_store = []
        self.bm25 = None

    # Add a document to the vector store
    def add_document(self, doc_id: str, text: str, paper: str = "generic", chunk_id: int = 0):
        text = (text or "").strip()
        # Empty / whitespace-only chunks break BM25Okapi (ZeroDivisionError in _calc_idf when idf is empty).
        if not text:
            return
        embedding = self.model.encode(text).tolist()
        metadata = {
            "paper": paper,
            "chunk_id": chunk_id
        }
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        # BM25 index: corpus[i], tokenized_corpus[i], metadata_store[i] must stay aligned.
        token = text.lower().split() or ["_"]
        self.corpus.append(text)
        self.tokenized_corpus.append(token)
        self.metadata_store.append(metadata)
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except (ZeroDivisionError, ValueError):
            # Corpus edge cases or library quirks (e.g. empty vocabulary); hybrid_search still has vector leg.
            self.bm25 = None

    # Add multiple documents to the vector store
    def add_documents(self, documents: list, paper: str):
        chunk_id = 0
        for text in documents:
            if not (text or "").strip():
                continue
            doc_id = f"{paper}_{chunk_id}"
            self.add_document(doc_id, text, paper, chunk_id)
            chunk_id += 1

    # Search for similar documents based on a query
    def search(self, query: str, n_results: int = 3):
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        documents = results["documents"][0]
        scores = results["distances"][0]
        metadatas = results["metadatas"][0]

        sources = []

        for doc, score, meta in zip(documents, scores, metadatas):
            sources.append({
                "text": doc,
                "score": score,
                "paper": meta["paper"],
                "chunk_id": meta["chunk_id"]
            })
        return sources
    
    # Search for relevant documents using keyword matching (BM25)
    def keyword_search(self, query: str, n_results: int = 3):
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # argsort is ascending; take last n indices then reverse so highest BM25 scores first
        top_indices = argsort(bm25_scores)[-n_results:][::-1]
        results = []
        # We retrieve the original documents from the corpus using the top indices and return them with their BM25 scores
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx],
                "score": bm25_scores[idx],
                "paper": self.metadata_store[idx]["paper"],
                "chunk_id": self.metadata_store[idx]["chunk_id"]
                })
        return results

    # Combine vector search and keyword search results
    def hybrid_search(self, query: str, n_results: int = 3):
        vector_results = self.search(query, n_results=5)
        keyword_results = self.keyword_search(query, n_results=5)
        combined_results = vector_results + keyword_results
    
        # We remove duplicates while preserving order, ensuring that we return a unique set of results based on the document text
        unique = {}
        for r in combined_results:
            key = r["text"]
            if key not in unique:
                unique[key] = r

        return list(unique.values())[:n_results]



