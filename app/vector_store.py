"""In-Memory Vector Store for Farming Schemes.

Performs vector indexing and sub-millisecond Cosine Similarity search using NumPy dot product.

Why In-Memory NumPy instead of an external Vector DB (Pinecone/Milvus/Qdrant)?
1. For 10-100 scheme documents (~50-200 passages), an in-memory numpy matrix dot product:
       scores = np.dot(embedding_matrix, query_vector)
   executes in ~0.1 - 0.3 milliseconds on CPU using optimized BLAS routines.
2. Zero network hops, zero container/daemon overhead, zero database cold start,
   and 100% self-contained reproducibility for deployment.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel

from app.embeddings import embedding_service
from app.schemes_data import FARMING_SCHEMES
from app.config import settings

logger = logging.getLogger("krishisahyak.vector_store")


class SearchResult(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    chunk_type: str
    content: str
    score: float
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self._build_index()

    def _build_index(self):
        """Processes scheme documents into granular semantic chunks and precomputes their embeddings."""
        logger.info("Building in-memory vector index from farming schemes...")
        raw_chunks = []

        for scheme in FARMING_SCHEMES:
            # 1. Scheme Summary Chunk
            raw_chunks.append({
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "category": scheme["category"],
                "chunk_type": "Overview",
                "text": f"Scheme: {scheme['name']} ({scheme['short_name']}). Category: {scheme['category']}. Ministry: {scheme['ministry']}. Summary: {scheme['full_text']} Highlights: {scheme['key_highlights']}",
                "metadata": {
                    "ministry": scheme["ministry"],
                    "short_name": scheme["short_name"],
                    "category": scheme["category"]
                }
            })

            # 2. Financial Benefit & Subsidy Chunk
            raw_chunks.append({
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "category": scheme["category"],
                "chunk_type": "Financial Assistance",
                "text": f"Financial Assistance and Subsidies for {scheme['name']}: {scheme['financial_assistance']}",
                "metadata": {
                    "financial_assistance": scheme["financial_assistance"],
                    "short_name": scheme["short_name"]
                }
            })

            # 3. Eligibility & Criteria Chunk
            raw_chunks.append({
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "category": scheme["category"],
                "chunk_type": "Eligibility",
                "text": f"Eligibility Criteria, Landholding, and Beneficiary Exclusions for {scheme['name']}: {scheme['eligibility']}",
                "metadata": {
                    "eligibility": scheme["eligibility"],
                    "short_name": scheme["short_name"]
                }
            })

            # 4. Application Process & Documents Chunk
            docs_str = ", ".join(scheme.get("documents_required", []))
            raw_chunks.append({
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "category": scheme["category"],
                "chunk_type": "Application Procedure",
                "text": f"How to Apply for {scheme['name']}: {scheme['application_procedure']}. Mandatory Documents Required: {docs_str}",
                "metadata": {
                    "application_procedure": scheme["application_procedure"],
                    "documents_required": scheme.get("documents_required", [])
                }
            })

        self.chunks = raw_chunks
        texts_to_embed = [c["text"] for c in self.chunks]

        logger.info(f"Encoding {len(texts_to_embed)} semantic chunks using Sentence-Transformers...")
        self.embedding_matrix = embedding_service.encode_batch(texts_to_embed)
        logger.info(f"Vector index created successfully with shape: {self.embedding_matrix.shape}")

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.20
    ) -> List[SearchResult]:
        """Performs Cosine Similarity search against all precomputed scheme vectors.
        
        Math Rationale:
        Since both self.embedding_matrix rows and query_vec are L2-normalized:
            similarity(query, doc_i) = np.dot(doc_i, query)
        Matrix multiplication:
            scores = np.dot(embedding_matrix, query_vec)
        gives all cosine similarities in a single vector operation.
        """
        if self.embedding_matrix is None or len(self.chunks) == 0:
            return []

        # 1. Embed incoming query
        query_vec = embedding_service.encode_text(query)

        # 2. Vectorized Cosine Similarity Dot Product
        # Shape: (N, 384) . (384,) -> (N,)
        scores = np.dot(self.embedding_matrix, query_vec)

        # 3. Sort by similarity score descending
        top_indices = np.argsort(scores)[::-1]

        results: List[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold and len(results) >= 1:
                # If we already have at least 1 match and subsequent scores drop below threshold, break
                break

            chunk_info = self.chunks[idx]
            results.append(SearchResult(
                scheme_id=chunk_info["scheme_id"],
                scheme_name=chunk_info["scheme_name"],
                category=chunk_info["category"],
                chunk_type=chunk_info["chunk_type"],
                content=chunk_info["text"],
                score=round(score, 4),
                metadata=chunk_info["metadata"]
            ))

            if len(results) >= top_k:
                break

        return results

    def get_all_schemes(self) -> List[Dict[str, Any]]:
        """Returns the full list of raw farming schemes."""
        return FARMING_SCHEMES


# Global singleton instance
vector_store = InMemoryVectorStore()
