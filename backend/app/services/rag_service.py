"""RAG Service — Retrieval-Augmented Generation for chat."""
from typing import Tuple, List
from loguru import logger
from app.services.index_service import IndexService
from app.core.config import settings

_qa_cache = None


def _get_qa_model():
    global _qa_cache
    if _qa_cache:
        return _qa_cache
    try:
        from transformers import pipeline
        _qa_cache = pipeline("text2text-generation", model=settings.qa_model, device=-1)
        return _qa_cache
    except Exception as e:
        logger.warning(f"QA model load failed: {e}")
        return None


class RAGService:
    @staticmethod
    async def answer(project_id: str, question: str, include_citations: bool = True) -> Tuple[str, List]:
        results = IndexService.search(project_id, question, top_k=5)
        if not results:
            return "I couldn't find relevant information in the video for that question.", []

        context = "\n\n".join(
            f"[{r['source']} @ {r.get('timestamp','')}] {r['text']}"
            for r in results
        )
        prompt = (
            f"You are a helpful AI assistant answering questions about a video.\n"
            f"Use ONLY the provided context. Cite timestamps when relevant.\n"
            f"If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        answer = RAGService._generate(prompt)
        citations = []
        if include_citations:
            for r in results[:3]:
                if r.get("timestamp"):
                    citations.append({
                        "timestamp": r["timestamp"],
                        "text": r["text"][:120] + "…",
                        "source": r["source"],
                    })
        return answer, citations

    @staticmethod
    def _generate(prompt: str) -> str:
        try:
            pipe = _get_qa_model()
            if pipe:
                result = pipe(prompt, max_new_tokens=300, do_sample=False)
                text = result[0].get("generated_text", "")
                if "Answer:" in text:
                    text = text.split("Answer:")[-1].strip()
                return text
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")
        # Fallback: return top result
        return "Based on the video content: " + prompt.split("Context:")[-1].split("Question:")[0].strip()[:300]
