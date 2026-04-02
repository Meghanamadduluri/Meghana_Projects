from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.schemas.schema import TextRequest, WordCountResponse, IngestRequest, SearchRequest, QueryRequest, RAGResponse
from app.services.async_examples import fake_embedding, fake_db_search
from app.db.vector_store import VectorStore
from app.services.generator import Generator
from app.services.document_loader import load_and_chunk_pdf

import logging

router = APIRouter()
vector_store = VectorStore()
generator = Generator()

#logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#dependency function to get settings
def get_settings():
    return {"app_name": "learning_api"} 

@router.post("/count_words" , response_model=WordCountResponse) 
async def count_words(
    req: TextRequest,
    settings: dict = Depends(get_settings)
):
    try:
        logger.info("Received text for word count") #logging the request
        words = req.text.split()
        count = len(words)
        if count > req.max_words:
            raise HTTPException(400, "text exceeds max_words")
        logger.info(f"Word count: {count}")
        return{"word_count": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return{"word_count": 0}
    
@router.post("/query")
async def query(req: TextRequest):              # wiring schema to endpoint, converting JSON into a python object
    embedding = await fake_embedding(req.text)
    results = await fake_db_search(embedding)
    return {"results": results}


@router.post("/ingest")
async def ingest_documents(req: IngestRequest):
    # 'req' contains 'documents' (a list), not 'text'
    for i, doc_text in enumerate(req.documents):
        # We generate a unique ID for each document in the list
        doc_id = f"doc_{i}" 
        vector_store.add_document(
            doc_id=doc_id,
            text=doc_text,
            paper="manual_ingest",
            chunk_id=i
        )
    
    return {"message": f"Successfully ingested {len(req.documents)} documents"}

@router.post("/search")
async def search_documents(req: SearchRequest):  
    results = vector_store.search(req.query) 
    return {
        "query": req.query,
        "results": results
    }

@router.post("/rag", response_model=RAGResponse)
async def rag_query(req: QueryRequest):
    sources = vector_store.hybrid_search(req.text, n_results=req.top_k)
    context = [s["text"] for s in sources]
    answer = generator.generate(req.text, context)

    return {
        "query": req.text,
        "answer": answer,
        "context": context,
    }


@router.post("/ingest_pdf")
async def ingest_pdf(file:UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    chunks = load_and_chunk_pdf(file_location)

    vector_store.add_documents(chunks, paper=file.filename)

    return {
        "chunks_added": len(chunks)
    }
