"""LLM Reasoning and Generation service using Groq API (Llama-3.3/3.1).

Features:
- Empathetic prompt engineering tailored for comprehensive farmer assistance without policy jargon.
- Intelligent conversational fallback synthesizer when GROQ_API_KEY is not configured yet.
- Sub-second token generation leveraging Groq LPU hardware.
"""

import logging
import re
import os
from typing import List, Dict, Any
from app.config import settings
from app.vector_store import SearchResult
from app.schemes_data import FARMING_SCHEMES

logger = logging.getLogger("krishisahyak.llm_service")


def clean_thinking_tags(text: str) -> str:
    """Strips internal <think>...</think> reasoning blocks from model outputs."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


class LLMService:
    def __init__(self):
        self.api_key = ""
        self.model = ""
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes or re-initializes the Groq client from settings/environment."""
        # Refresh from env / settings
        self.api_key = os.getenv("GROQ_API_KEY", settings.GROQ_API_KEY).strip()
        self.model = os.getenv("GROQ_MODEL", settings.GROQ_MODEL).strip() or "openai/gpt-oss-20b"

        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info(f"Groq LLM Client initialized with default model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self._client = None
        else:
            logger.warning("GROQ_API_KEY not found in environment. Running in Smart Context Synthesizer / Fallback mode.")
            self._client = None

    def build_prompt(self, question: str, retrieved_chunks: List[SearchResult]) -> tuple[str, str]:
        """Constructs a grounded system and user prompt with injected scheme contexts."""
        
        system_prompt = (
            "You are KrishiSahyak (कृषि सहायक), an empathetic, expert agricultural advisor dedicated to helping Indian farmers "
            "understand government schemes easily without getting lost in complicated policy jargon.\n\n"
            "IMPORTANT: Do NOT output <think> tags or internal thoughts. Respond directly to the farmer.\n"
            "FORMATTING GUIDELINE: Ensure every markdown table row is separated by a proper newline (`\\n`). Avoid squeezing multiple table rows into one line.\n\n"
            "YOUR GUIDING PRINCIPLES:\n"
            "1. DO NOT simply repeat or dump policy text. Explain the scheme conversationally, warmly, and clearly like a personal assistant.\n"
            "2. Provide complete, end-to-end guidance covering:\n"
            "   - 💡 **Direct Overview & Exact Benefits**: Clear summary of what the farmer receives (money, subsidy %, interest rate).\n"
            "   - 🎯 **Who is Eligible & Who is Excluded**: Simple criteria so the farmer immediately knows if they qualify.\n"
            "   - 📋 **Required Documents Checklist**: Clear list of papers needed.\n"
            "   - 🚀 **Step-by-Step Action Plan**: Exactly where to go (CSC, Portal, Bank, District Agriculture Office) and what to do.\n"
            "   - 💡 **Farmer Advisory Tip**: Practical tips (e.g. e-KYC, bank-Aadhaar linking, cut-off deadlines).\n"
            "3. Ground all factual details (amounts, percentages, document names) strictly in the provided Context below.\n"
            "4. If the context lacks details for a specific question, explain what is available and guide the farmer to their nearest Krishi Vigyan Kendra (KVK) or District Agriculture Office."
        )

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source {i}: {chunk.scheme_name} | {chunk.category} | {chunk.chunk_type}]\n{chunk.content}"
            )

        context_text = "\n\n".join(context_blocks)

        user_prompt = (
            f"=== RETRIEVED SCHEME CONTEXTS ===\n"
            f"{context_text}\n"
            f"=================================\n\n"
            f"FARMER QUESTION: {question}\n\n"
            f"Please provide a complete, warm, and step-by-step assistant explanation based on the context above:"
        )

        return system_prompt, user_prompt

    def generate_answer(self, question: str, retrieved_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Generates an answer using Groq API or falls back to structured context synthesis."""
        if not retrieved_chunks:
            return {
                "answer": "No relevant agricultural schemes matched your question. Please try rephrasing or search for specific terms like 'PM-KISAN', 'crop insurance', 'solar pump', or 'organic farming'.",
                "model_used": "none",
                "citations": []
            }

        citations = [
            {
                "scheme_id": c.scheme_id,
                "scheme_name": c.scheme_name,
                "category": c.category,
                "chunk_type": c.chunk_type,
                "relevance_score": c.score
            }
            for c in retrieved_chunks
        ]

        system_prompt, user_prompt = self.build_prompt(question, retrieved_chunks)

        # Ensure client is initialized if environment variable was added dynamically
        if self._client is None:
            self._init_client()

        # 1. Groq LLM Inference with model fallback list
        if self._client is not None:
            # Candidate models list starting with user-configured model
            candidate_models = [self.model, "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]
            dedup_candidates = []
            for m in candidate_models:
                if m and m not in dedup_candidates:
                    dedup_candidates.append(m)

            for target_model in dedup_candidates:
                try:
                    response = self._client.chat.completions.create(
                        model=target_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,  # Low temperature for factual, grounded retrieval
                        max_tokens=768,   # Stay under 1000 token limit to avoid OTPM rate limit errors
                        top_p=0.9
                    )
                    raw_answer = response.choices[0].message.content
                    answer_text = clean_thinking_tags(raw_answer)

                    return {
                        "answer": answer_text,
                        "model_used": f"Groq/{target_model}",
                        "citations": citations
                    }
                except Exception as e:
                    logger.warning(f"Groq API call failed for model '{target_model}': {e}")
                    continue

        # 2. Smart Context Synthesis Fallback (when GROQ_API_KEY is not yet added or API fails)
        fallback_answer = self._generate_fallback_synthesis(question, retrieved_chunks)
        return {
            "answer": fallback_answer,
            "model_used": "local-context-synthesizer (Check GROQ_API_KEY in .env for Groq generation)",
            "citations": citations
        }

    def _generate_fallback_synthesis(self, question: str, chunks: List[SearchResult]) -> str:
        """Synthesizes an empowering, comprehensive advisory response from scheme data when Groq key is absent."""
        primary_match = chunks[0]
        scheme_id = primary_match.scheme_id
        
        # Find full scheme metadata from dataset
        scheme_data = next((s for s in FARMING_SCHEMES if s["id"].lower() == scheme_id.lower()), None)
        
        lines = []
        lines.append(f"### 🌾 **KrishiSahyak Advisory Guide: {primary_match.scheme_name}**")
        if scheme_data:
            lines.append(f"*(Category: {primary_match.category} | Ministry: {scheme_data.get('ministry', 'Govt of India')})*")
        lines.append("")
        lines.append("Namaste! Here is your complete, step-by-step guidance so you don't have to figure out complex policy documents yourself:")
        lines.append("")

        if scheme_data:
            lines.append("### 💰 **1. Financial Benefits & Subsidies**")
            lines.append(f"{scheme_data['financial_assistance']}")
            lines.append("")
            
            lines.append("### 🎯 **2. Who is Eligible & Key Criteria**")
            lines.append(f"{scheme_data['eligibility']}")
            lines.append("")
            
            lines.append("### 📋 **3. Mandatory Documents Required Checklist**")
            for doc in scheme_data.get("documents_required", []):
                lines.append(f"- [ ] **{doc}**")
            lines.append("")

            lines.append("### 🚀 **4. Step-by-Step Application Guide**")
            lines.append(f"{scheme_data['application_procedure']}")
            lines.append("")

            lines.append("### 💡 **5. Farmer Advisory & Practical Tips**")
            lines.append(f"• **Key Highlights:** {scheme_data['key_highlights']}")
            lines.append("• **Action Tip:** Ensure your Aadhaar is linked to your active bank account and NPCI Direct Benefit Transfer (DBT) is enabled for smooth credit.")
            lines.append("")
        else:
            # Fallback if metadata lookup fails
            lines.append("### 💡 **Key Information & Benefits**")
            lines.append(primary_match.content)
            lines.append("")

        if len(chunks) > 1:
            lines.append("---")
            lines.append("### 🔍 **Other Related Schemes You Might Benefit From:**")
            for c in chunks[1:]:
                other_scheme = next((s for s in FARMING_SCHEMES if s["id"].lower() == c.scheme_id.lower()), None)
                benefit = other_scheme['financial_assistance'] if other_scheme else c.content[:200]
                lines.append(f"- **{c.scheme_name}** ({c.category}): {benefit}")
                lines.append("")

        lines.append("> *Note: Set `GROQ_API_KEY` in `.env` to activate full Groq AI conversational reasoning.*")
        return "\n".join(lines)


# Global singleton instance
llm_service = LLMService()
