from pydantic import BaseModel, Field
from typing import List

class TextRequest(BaseModel): 
    text: str = Field(..., min_length=1, max_length=1000)
    max_words: int = Field(...,ge=1, le=500)
    
class WordCountResponse(BaseModel): 
    word_count: int 

class IngestRequest(BaseModel):
    documents: List[str]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)

class QueryResponse(BaseModel):
    query: str
    results: list[str]

class RAGResponse(BaseModel):
    query: str
    answer: str
    context: list[str]

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3