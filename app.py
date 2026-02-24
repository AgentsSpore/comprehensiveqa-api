"""ComprehensiveQA API - Retrieve-Verify-Retrieve Framework"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from rvr_engine import RVREngine
from document_store import DocumentStore

app = FastAPI(
    title="ComprehensiveQA API",
    description="Multi-round retrieval for comprehensive question answering",
    version="1.0.0"
)

# Initialize document store and RVR engine
doc_store = DocumentStore()
rvr_engine = RVREngine(doc_store)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to answer comprehensively")
    max_rounds: int = Field(3, ge=1, le=5, description="Maximum retrieval rounds")
    docs_per_round: int = Field(5, ge=1, le=10, description="Documents per round")
    verification_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum relevance score")


class DocumentResult(BaseModel):
    text: str
    score: float
    round: int
    source: str


class QueryResponse(BaseModel):
    query: str
    total_rounds: int
    verified_documents: List[DocumentResult]
    coverage_improvement: str


@app.get("/")
async def root():
    return {
        "service": "ComprehensiveQA API",
        "paper": "https://arxiv.org/abs/2602.18425v1",
        "description": "Retrieve-Verify-Retrieve for comprehensive answer coverage"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute multi-round RVR retrieval for comprehensive answers"""
    try:
        results = rvr_engine.retrieve_verify_retrieve(
            query=request.query,
            max_rounds=request.max_rounds,
            docs_per_round=request.docs_per_round,
            verification_threshold=request.verification_threshold
        )
        
        verified_docs = [
            DocumentResult(
                text=doc["text"],
                score=doc["score"],
                round=doc["round"],
                source=doc["source"]
            )
            for doc in results["verified_documents"]
        ]
        
        return QueryResponse(
            query=request.query,
            total_rounds=results["total_rounds"],
            verified_documents=verified_docs,
            coverage_improvement=f"{results['coverage_improvement']:.1f}%"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Get document store statistics"""
    return {
        "total_documents": len(doc_store.documents),
        "categories": list(set(doc["category"] for doc in doc_store.documents))
    }


if __name__ == "__main__":
    print("🚀 Starting ComprehensiveQA API...")
    print("📄 Based on: https://arxiv.org/abs/2602.18425v1")
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)