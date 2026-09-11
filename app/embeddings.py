"""Embedding service using FastEmbed (ONNX Runtime backend).

Features:
- Singleton model initialization to avoid reloading heavy weights on every request.
- Manual L2 vector normalization so cosine similarity reduces to a fast dot product.
- Batch encoding support for precomputing scheme document vectors during startup.

Why FastEmbed instead of Sentence-Transformers:
Sentence-Transformers depends on PyTorch, whose memory footprint (300MB+ just to
import, before loading any model) exceeds free-tier hosting limits like Render's
512MB cap. FastEmbed uses ONNX Runtime instead, running the same underlying
all-MiniLM-L6-v2 model with a much smaller memory footprint, while producing
embeddings of the same shape and (near-identical) quality.
"""

import logging
from typing import List
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
        """Initializes the FastEmbed model."""
        try:
            from fastembed import TextEmbedding
            model_name = settings.EMBEDDING_MODEL_NAME
            if model_name == "all-MiniLM-L6-v2":
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Loading FastEmbed model: '{model_name}'...")
            self._model = TextEmbedding(model_name=model_name)
            logger.info("FastEmbed model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FastEmbed model: {e}. Falling back to deferred/mock mode.")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector so cosine similarity reduces to a dot product."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes a single text string into a 1D L2-normalized numpy vector (shape: [384])."""
        if self._model is not None:
            # FastEmbed's .embed() returns a generator of arrays, one per input text.
            # We pass a single-item list and take the first (and only) result.
            vector = list(self._model.embed([text]))[0]
            vector = np.array(vector, dtype=np.float32)
            return self._normalize(vector)

        # Fallback dummy embedding (384 dimensions) for offline/testing if model couldn't load
        logger.warning("Using fallback pseudo-random vector (FastEmbed not initialized).")
        pseudo_vec = np.random.randn(384).astype(np.float32)
        return pseudo_vec / np.linalg.norm(pseudo_vec)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text strings into a 2D numpy matrix of shape (N, 384)."""
        if self._model is not None:
            vectors = list(self._model.embed(texts))
            matrix = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.maximum(norms, 1e-12)

        # Fallback dummy matrix
        logger.warning("Using fallback pseudo-random matrix.")
        mat = np.random.randn(len(texts), 384).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-12)

# Singleton global instance
embedding_service = EmbeddingService()