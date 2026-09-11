"""Main FastAPI Application for KrishiSahyak (कृषि सहायक).

Provides:
- POST /ask : Semantic search + RAG pipeline with Groq LLM reasoning
- GET  /schemes : List all indexed agricultural schemes
- GET  /schemes/{scheme_id} : Get detailed scheme specification
- GET  /health : System status, vector index statistics, and model health
"""

import time
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings import embedding_service
from app.vector_store import vector_store, SearchResult
from app.llm_service import llm_service
from app.schemes_data import FARMING_SCHEMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("krishisahyak.main")

# Initialize FastAPI app with descriptive OpenAPI documentation
app = FastAPI(
    title="KrishiSahyak (कृषि सहायक) - Farming Schemes RAG API",
    description=(
        "An intelligent Retrieval-Augmented Generation (RAG) backend that assists farmers, "
        "agri-entrepreneurs, and field officers with instant, grounded answers about Indian Government "
        "agricultural schemes using Sentence-Transformers embeddings and Groq LLM inference."
    ),
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (Frontend UI)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Optional Conceptual Auth Scheme (Preserves protected route signature without DB overhead)
security = HTTPBearer(auto_error=False)


def verify_conceptual_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Dict[str, Any]:
    """Conceptual authentication dependency for protected routes.
    
    Accepts:
    - Bearer token in Authorization header (e.g. 'Bearer demo-token-123')
    - Or X-API-Key header
    - If neither is provided, allows anonymous demo access with a logged notice.
    """
    if credentials:
        token = credentials.credentials
        return {"authenticated": True, "auth_type": "Bearer", "user": "verified_farmer_user", "token": token}
    elif x_api_key:
        return {"authenticated": True, "auth_type": "APIKey", "user": "api_client"}
    
    # Allows frictionless testing in /docs while demonstrating the auth dependency pattern
    return {"authenticated": False, "auth_type": "demo", "user": "guest_farmer"}


# --- Request & Response Models ---

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Farmer's query regarding agricultural schemes, subsidies, insurance, or loans.",
        example="What financial assistance is provided under PM-KISAN and what are the eligibility criteria?"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top matching scheme passages to retrieve for context grounding."
    )
    min_similarity: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score threshold (0.0 - 1.0) to filter irrelevant documents."
    )


class Citation(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    chunk_type: str
    relevance_score: float


class RetrievedContext(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    chunk_type: str
    content: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    execution_time_ms: float
    retrieved_contexts: List[RetrievedContext]
    citations: List[Citation]
    sources: List[str] = Field(default_factory=list, description="Unique referenced scheme names")
    auth_context: Dict[str, Any]


# --- API Endpoints ---

@app.get("/", tags=["General"], response_class=HTMLResponse)
def root():
    """Serves the KrishiSahyak Frontend Dashboard or API information."""
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "endpoints": {
            "ask": "POST /ask",
            "schemes": "GET /schemes",
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint displaying vector index size, model status, and config."""
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimensions": 384,
        "indexed_schemes_count": len(FARMING_SCHEMES),
        "indexed_chunks_count": len(vector_store.chunks),
        "groq_configured": bool(settings.GROQ_API_KEY.strip()),
        "groq_model": settings.GROQ_MODEL
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG Question Answering"])
def ask_farming_scheme(
    request: AskRequest,
    auth: Dict[str, Any] = Depends(verify_conceptual_auth)
):
    """Core RAG Endpoint:
    
    1. Embeds incoming farmer question using Sentence-Transformers (all-MiniLM-L6-v2).
    2. Performs Cosine Similarity vector search against precomputed in-memory scheme chunks.
    3. Injects retrieved top-K context into a strictly grounded system prompt.
    4. Calls Groq API (Llama-3.3-70b) for ultra-fast, hallucination-free generation.
    5. Returns structured answer with source citations, relevance scores, and execution metrics.
    """
    start_time = time.perf_counter()

    # Step 1 & 2: Semantic search via vector store
    retrieved_results: List[SearchResult] = vector_store.search(
        query=request.question,
        top_k=request.top_k,
        threshold=request.min_similarity
    )

    # Step 3 & 4: LLM context injection & generation via Groq
    llm_output = llm_service.generate_answer(
        question=request.question,
        retrieved_chunks=retrieved_results
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Convert retrieved results to response models
    contexts = [
        RetrievedContext(
            scheme_id=r.scheme_id,
            scheme_name=r.scheme_name,
            category=r.category,
            chunk_type=r.chunk_type,
            content=r.content,
            score=r.score
        )
        for r in retrieved_results
    ]

    citations = [
        Citation(
            scheme_id=c["scheme_id"],
            scheme_name=c["scheme_name"],
            category=c["category"],
            chunk_type=c["chunk_type"],
            relevance_score=c["relevance_score"]
        )
        for c in llm_output["citations"]
    ]

    unique_sources = []
    for c in llm_output["citations"]:
        s_name = c["scheme_name"]
        if s_name not in unique_sources:
            unique_sources.append(s_name)

    return AskResponse(
        question=request.question,
        answer=llm_output["answer"],
        model_used=llm_output["model_used"],
        execution_time_ms=elapsed_ms,
        retrieved_contexts=contexts,
        citations=citations,
        sources=unique_sources,
        auth_context=auth
    )


@app.get("/schemes", tags=["Farming Schemes Database"])
def list_all_schemes():
    """Lists all indexed farming schemes with their categories and key benefits."""
    return {
        "total_schemes": len(FARMING_SCHEMES),
        "schemes": [
            {
                "id": s["id"],
                "name": s["name"],
                "short_name": s["short_name"],
                "category": s["category"],
                "ministry": s["ministry"],
                "financial_assistance": s["financial_assistance"]
            }
            for s in FARMING_SCHEMES
        ]
    }


@app.get("/schemes/{scheme_id}", tags=["Farming Schemes Database"])
def get_scheme_detail(scheme_id: str):
    """Returns detailed specification, eligibility, documents, and application procedure for a scheme."""
    for scheme in FARMING_SCHEMES:
        if scheme["id"].lower() == scheme_id.lower():
            return scheme
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Scheme with ID '{scheme_id}' not found. Check GET /schemes for available IDs."
    )
