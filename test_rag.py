"""Offline verification script for KrishiSahyak RAG Service."""

import sys
import os
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.embeddings import embedding_service
from app.vector_store import vector_store
from app.llm_service import llm_service
from app.schemes_data import FARMING_SCHEMES

def main():
    print("=" * 60)
    print("KrishiSahyak - Farming Schemes RAG Pipeline Verification")
    print("=" * 60)
    
    print(f"\n1. Total Schemes in Dataset: {len(FARMING_SCHEMES)}")
    for s in FARMING_SCHEMES[:5]:
        print(f"   - [{s['id']}] {s['name']} ({s['category']})")
    print("   ... and more.")

    print(f"\n2. Vector Store Indexed Chunks: {len(vector_store.chunks)}")
    print(f"   Embedding Matrix Shape: {vector_store.embedding_matrix.shape}")

    test_queries = [
        "How much money do farmers get under PM-KISAN and who is eligible?",
        "How can I get insurance compensation for crop loss due to drought or hailstorm?",
        "What subsidy is provided for installing drip irrigation systems?",
        "What are the interest rates and credit limit on Kisan Credit Card?",
        "How can I get financial assistance to buy a farm tractor or equipment?",
        "Is there any pension scheme for old age small and marginal farmers?"
    ]

    print("\n3. Testing Semantic Cosine Similarity Search:")
    for query in test_queries:
        print("\n" + "-" * 50)
        print(f"Query: '{query}'")
        t0 = time.perf_counter()
        results = vector_store.search(query, top_k=2, threshold=0.20)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Retrieval Time: {elapsed:.2f} ms")
        
        for idx, res in enumerate(results, 1):
            print(f"  Match #{idx}: {res.scheme_name} [{res.chunk_type}] (Score: {res.score:.4f})")

    print("\n4. Testing LLM Service Grounded Generation:")
    sample_query = "What financial assistance is provided under PM-KISAN and who is not eligible?"
    retrieved = vector_store.search(sample_query, top_k=2)
    response = llm_service.generate_answer(sample_query, retrieved)
    print(f"Model: {response['model_used']}")
    print(f"Answer Preview:\n{response['answer'][:400]}...")
    print(f"Citations: {[c['scheme_name'] for c in response['citations']]}")

    print("\n" + "=" * 60)
    print("Verification Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
