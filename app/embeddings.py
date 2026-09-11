"""Embedding service using Sentence-Transformers.

Features:
- Singleton model initialization to avoid reloading heavy weights on every request.
- Automatic L2 vector normalization so cosine similarity reduces to a fast dot product.
- Batch encoding support for precomputing scheme document vectors during startup.
"""

import logging
from typing import List, Union
import numpy as np
from app.config import settings

logger = logging.getLogger("krishisahyak.embeddings")

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Initializes the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading Sentence-Transformer model: '{settings.EMBEDDING_MODEL_NAME}'...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Sentence-Transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}. Falling back to deferred/mock mode.")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes a single text string into a 1D L2-normalized numpy vector (shape: [384]).
        
        Why normalize_embeddings=True?
        When vectors have unit norm (||v|| = 1), Cosine Similarity:
            cos(u, v) = (u . v) / (||u|| * ||v||) == u . v (simple dot product)
        This avoids expensive square root and norm computations during search queries.
        """
        if self._model is not None:
            vector = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return np.array(vector, dtype=np.float32)
        
        # Fallback dummy embedding (384 dimensions) for offline/testing if model couldn't load
        logger.warning("Using fallback pseudo-random vector (SentenceTransformer not initialized).")
        pseudo_vec = np.random.randn(384).astype(np.float32)
        return pseudo_vec / np.linalg.norm(pseudo_vec)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text strings into a 2D numpy matrix of shape (N, 384)."""
        if self._model is not None:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            return np.array(vectors, dtype=np.float32)
        
        # Fallback dummy matrix
        logger.warning("Using fallback pseudo-random matrix.")
        mat = np.random.randn(len(texts), 384).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-12)

# Singleton global instance
embedding_service = EmbeddingService()
