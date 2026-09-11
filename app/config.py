import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "KrishiSahyak (कृषि सहायक) - Farming Schemes RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Embedding Configuration
    # 'all-MiniLM-L6-v2': 384 dimensions, fast CPU inference (<15ms), ~80MB footprint
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Groq LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    
    # RAG Retrieval Configuration
    DEFAULT_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.25  # Minimum cosine similarity to consider relevant

settings = Settings()
