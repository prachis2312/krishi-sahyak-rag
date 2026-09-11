# KrishiSahyak (कृषि सहायक) - Farming Schemes RAG Service

An intelligent Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **Sentence-Transformers**, **In-Memory Cosine Similarity Vector Store (NumPy)**, and **Groq LLM (Llama 3.3)**.

Provides instant, grounded, and hallucination-free guidance on 14+ Indian Government agricultural schemes, credit policies, crop insurance, and subsidies.

---

## 🌟 Key Features

1. **Dense Semantic Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) to understand natural language intent beyond keyword matching (e.g. searching *"money for drought damage"* accurately retrieves *PM Fasal Bima Yojana*).
2. **In-Memory Vector Search**: Sub-millisecond Cosine Similarity search ($\vec{q} \cdot D^T$) via vectorized NumPy matrix multiplication.
3. **Context Grounded LLM**: Integrates with Groq Cloud API (`llama-3.3-70b-versatile`) with strict system prompts that eliminate hallucination and cite official scheme names and criteria.
4. **Interactive OpenAPI / Swagger UI**: Built-in interactive documentation and test sandbox available at `/docs`.
5. **Conceptual Auth Header**: Implements FastAPI dependency injection (`HTTPBearer`) for route protection patterns.

---

## 📐 Architecture & Function Choice Rationale (Interview & Defense Guide)

| Component | Function / Class | Why This Was Chosen |
|---|---|---|
| **Embedding** | `SentenceTransformer.encode(texts, normalize_embeddings=True)` in `app/embeddings.py` | L2 vector normalization ensures $\|\vec{v}\| = 1$. This simplifies Cosine Similarity $\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|\|\vec{v}\|}$ to a fast single dot product $\vec{u} \cdot \vec{v}$. |
| **Vector Search** | `np.dot(embedding_matrix, query_vec)` in `app/vector_store.py` | For small-to-medium corpora (10-100 scheme docs / 50-200 passages), in-memory NumPy runs in $<0.3\text{ ms}$ on CPU, eliminating database network hops and vector DB daemon overhead. |
| **LLM Inference** | `Groq.chat.completions.create(...)` in `app/llm_service.py` | Groq LPU hardware achieves 300–500 tokens/sec for sub-second RAG response times with strong reasoning and citation formatting. |
| **Auth Dependency** | `Depends(verify_conceptual_auth)` in `app/main.py` | Demonstrates the FastAPI dependency injection security pattern (`OAuth2PasswordBearer` / `HTTPBearer`) without heavy database migrations. |

---

## 📂 Project Structure

```
KrishiSahyak/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment settings & configuration
│   ├── schemes_data.py        # 14 Curated Indian Farming Schemes with full metadata
│   ├── embeddings.py          # Sentence-Transformers singleton loader & normalizer
│   ├── vector_store.py        # In-Memory Cosine Similarity Vector Index (NumPy)
│   ├── llm_service.py         # Groq LLM client (Llama-3.3) with fallback synthesizer
│   └── main.py                # FastAPI routes (/ask, /schemes, /health) & OpenAPI docs
├── .env.example               # Environment variables template
├── .env                       # Local environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Configure Groq API Key (Optional but Recommended)
Get a free API key from [Groq Console](https://console.groq.com/keys) and add it to your `.env` file:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(Note: If you run without a key, KrishiSahyak will automatically use its Smart Context Synthesizer mode so all vector searches and citations work!)*

### 2. Start the FastAPI Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open Swagger UI
Navigate in your browser to:
👉 **`http://localhost:8000/docs`**

---

## 🧪 Testing the `/ask` Endpoint

### Example 1: PM-KISAN Benefit
**Question:** *"How much money do farmers get under PM-KISAN and what is the eligibility?"*
- **Retrieved Scheme:** `Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)`
- **Answer Highlights:** Rs. 6,000 per year in 3 equal installments of Rs. 2,000 via DBT. Ineligibility includes institutional landholders, government employees, and income-tax payers.

### Example 2: Crop Loss Insurance
**Question:** *"How can I get insurance compensation if my crop is damaged due to drought or hailstorm?"*
- **Retrieved Scheme:** `Pradhan Mantri Fasal Bima Yojana (PMFBY)`
- **Answer Highlights:** Nominal premium rates (2% Kharif, 1.5% Rabi, 5% Commercial/Horticulture), covers prevented sowing, standing crop damage, and post-harvest losses.

### Example 3: Micro-Irrigation / Drip Subsidy
**Question:** *"What subsidy is available for installing drip irrigation?"*
- **Retrieved Scheme:** `PMKSY - Per Drop More Crop (PMKSY-PDMC)`
- **Answer Highlights:** 55% subsidy for Small/Marginal farmers (<2 hectares) and 45% for other farmers.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question about farming schemes (RAG search + Groq LLM) |
| `GET` | `/schemes` | Get a list of all indexed farming schemes |
| `GET` | `/schemes/{id}` | Get complete details and application steps for a scheme |
| `GET` | `/health` | System status, vector index chunk count, and model diagnostics |
| `GET` | `/docs` | Interactive OpenAPI / Swagger UI |
