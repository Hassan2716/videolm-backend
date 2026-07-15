"""
RAG v2 — TF-IDF retrieval + Groq API (Llama 3.1 8B).

Chatbot fixes:
- Reads GROQ_API_KEY directly from os.environ (not settings) to avoid venv issues
- Strong anti-hallucination system prompt — forces grounding in context ONLY
- Removes repetitive output from FLAN-T5 fallback
- Shorter context window to prevent token overflow repetition
"""
import os, re, pickle
from typing import List, Dict, Tuple, Optional
from loguru import logger
from app.core.config import settings

_indexes: Dict = {}

# ── TF-IDF Retriever ──────────────────────────────────────────────────────────
class TFIDFRetriever:
    def __init__(self):
        self.docs = []; self.mat = None; self.vec = None

    def build(self, docs):
        self.docs = docs
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1,2))
            self.mat = self.vec.fit_transform([d["text"] for d in docs])
            logger.info(f"TF-IDF built: {len(docs)} docs")
        except Exception as e:
            logger.warning(f"TF-IDF build failed: {e}")

    def search(self, query, top_k=5):
        if self.vec is None: return self._kw(query, top_k)
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            s = cosine_similarity(self.vec.transform([query]), self.mat).flatten()
            idxs = np.argsort(s)[::-1][:top_k]
            return [{**self.docs[i], "score": float(s[i])} for i in idxs if s[i] > 0.01]
        except Exception:
            return self._kw(query, top_k)

    def _kw(self, query, top_k):
        qw = set(re.findall(r'\b\w{3,}\b', query.lower()))
        scored = []
        for d in self.docs:
            dw = set(re.findall(r'\b\w{3,}\b', d["text"].lower()))
            overlap = len(qw & dw)
            if overlap > 0:
                scored.append({**d, "score": overlap / max(len(qw), 1)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def _ts(s) -> str:
    try:
        s = float(s)
        h, m, sec = int(s//3600), int((s%3600)//60), int(s%60)
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
    except: return "00:00"


def _chunk_segs(segs, target=150, overlap=20):
    result, cur, cs, ce = [], [], None, 0
    for seg in segs:
        w = seg.get("text","").split()
        if cs is None: cs = seg.get("start", 0)
        cur.extend(w); ce = seg.get("end", cs)
        if len(cur) >= target:
            result.append({"text": " ".join(cur), "start": cs, "end": ce})
            cur = cur[-overlap:]; cs = seg.get("start", ce)
    if len(cur) > 20:
        result.append({"text": " ".join(cur), "start": cs, "end": ce})
    return result


def build_index(project_id, full_text, segments, frames, chunks=None):
    docs = []
    seg_chunks = _chunk_segs(segments, 150, 20) if segments else []
    if seg_chunks:
        for c in seg_chunks:
            docs.append({"text": c["text"], "source": "transcript",
                          "start": c["start"], "end": c["end"], "timestamp": _ts(c["start"])})
    else:
        words = full_text.split()
        for i in range(0, len(words), 150):
            docs.append({"text": " ".join(words[i:i+150]), "source": "transcript",
                          "start": None, "end": None, "timestamp": None})
    for f in frames:
        if f.get("caption") and len(f["caption"]) > 15:
            docs.append({"text": f["caption"], "source": "frame_caption",
                          "start": f.get("ts",0), "end": f.get("ts",0), "timestamp": _ts(f.get("ts",0))})
        if f.get("ocr") and len(f["ocr"]) > 20:
            docs.append({"text": f["ocr"], "source": "slide_text",
                          "start": f.get("ts",0), "end": f.get("ts",0), "timestamp": _ts(f.get("ts",0))})
    if not docs: return

    r = TFIDFRetriever(); r.build(docs); _indexes[project_id] = r
    path = os.path.join(settings.faiss_index_dir, project_id)
    os.makedirs(path, exist_ok=True)
    try:
        with open(os.path.join(path, "tfidf.pkl"), "wb") as f:
            pickle.dump({"docs": r.docs, "vec": r.vec, "mat": r.mat}, f)
    except Exception as e:
        logger.warning(f"Index save failed: {e}")


def retrieve(project_id, query, top_k=5):
    r = _indexes.get(project_id)
    if r is None:
        path = os.path.join(settings.faiss_index_dir, project_id, "tfidf.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f: data = pickle.load(f)
                r = TFIDFRetriever()
                r.docs = data["docs"]
                r.vec = data.get("vec") or data.get("vectorizer")
                r.mat = data.get("mat") or data.get("tfidf_matrix")
                _indexes[project_id] = r
            except Exception as e:
                logger.warning(f"Index load failed: {e}")
    if r is None: return _db_fallback(project_id, query, top_k)
    return r.search(query, top_k)


def _db_fallback(project_id, query, top_k):
    try:
        from app.core.database import SessionLocal, Transcript
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
        db.close()
        if not t or not t.full_text: return []
        sents = re.split(r'(?<=[.!?])\s+', t.full_text)
        qw = set(re.findall(r'\b\w{3,}\b', query.lower()))
        scored = []
        for s in sents:
            ow = len(qw & set(re.findall(r'\b\w{3,}\b', s.lower())))
            if ow > 0:
                scored.append({"text": s, "source": "transcript",
                                "start": None, "timestamp": None,
                                "score": ow / max(len(qw), 1)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    except Exception as e:
        logger.error(f"DB fallback: {e}"); return []


# ── Anti-hallucination system prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an educational AI assistant for VideoLM. You ONLY answer based on the video transcript excerpts provided.

STRICT RULES:
1. ONLY use information from the Context section below. NEVER use your own training knowledge.
2. If the answer is not in the context, say exactly: "This specific question wasn't covered in the video. Based on what was discussed: [mention what IS in the context]"
3. Keep answers concise and clear — 2-5 sentences maximum unless asked for detail
4. Always cite timestamps like [02:15] when available in the context
5. Never repeat the same phrase more than once
6. Never list concepts comma-separated — always use full sentences"""


async def generate_answer(
    project_id: str,
    question: str,
    include_citations: bool = True,
    history: Optional[List[Dict]] = None,
) -> Tuple[str, List[Dict]]:
    results = retrieve(project_id, question, top_k=5)
    if not results:
        return ("I couldn't find relevant content in the video for that question. "
                "Make sure the video has been fully processed and try rephrasing.", [])

    # Build context — keep it SHORT to avoid token overflow / repetition
    context_parts = []
    for r in results[:4]:
        ts = f"[{r['timestamp']}] " if r.get("timestamp") else ""
        context_parts.append(f"{ts}{r['text'][:300]}")
    context = "\n\n".join(context_parts)

    answer = await _call_groq(question, context, history or [])
    answer = _clean_answer(answer)

    citations = []
    if include_citations:
        for r in results[:3]:
            if r.get("timestamp"):
                citations.append({
                    "timestamp": r["timestamp"],
                    "text": r["text"][:150],
                    "source": r.get("source", "transcript"),
                    "score": round(r.get("score", 0), 3),
                })
    return answer, citations


def _clean_answer(text: str) -> str:
    """Remove repetition artifacts common in FLAN-T5 output."""
    if not text: return "I couldn't generate an answer. Please try again."
    # Remove repeated phrases (the "naive-based classifier, naive-based..." problem)
    text = re.sub(r'\b(\w[\w\s-]{2,30})(,\s*\1){3,}', r'\1', text)
    # Remove repeated single words
    text = re.sub(r'\b(\w+)(\s+\1){3,}', r'\1', text)
    # Truncate if still very long (> 600 chars) at last sentence boundary
    if len(text) > 600:
        last = max(text[:600].rfind('.'), text[:600].rfind('!'), text[:600].rfind('?'))
        if last > 200: text = text[:last+1]
    return text.strip()


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"


async def _call_groq(question: str, context: str, history: List[Dict]) -> str:
    """Call Groq. Reads API key directly from env (bypasses venv/settings issue)."""
    import httpx

    # Read directly from os.environ — not settings — to avoid venv env var loading issues
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if api_key and api_key != "your_groq_api_key_here":
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Last 3 turns only to keep context small
            for h in (history[-6:] if history else []):
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"][:300]})

            messages.append({
                "role": "user",
                "content": f"VIDEO CONTEXT (use ONLY this):\n{context}\n\nQUESTION: {question}"
            })

            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "max_tokens": 400,
                        "temperature": 0.1,  # very low — reduces hallucination
                        "top_p": 0.9,
                    },
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    logger.info(f"Groq answered: {len(content)} chars")
                    return content
                else:
                    logger.warning(f"Groq error {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")
    else:
        logger.warning("GROQ_API_KEY not set or is placeholder — using FLAN-T5 fallback")

    return _flan_answer(question, context)


def _flan_answer(question: str, context: str) -> str:
    """FLAN-T5 fallback with strict prompt to prevent repetition."""
    try:
        from transformers import pipeline
        pipe = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        # Very constrained prompt — forces short answer
        prompt = (
            f"Answer in 1-2 sentences using ONLY this context. Do not repeat words.\n"
            f"Context: {context[:800]}\n"
            f"Question: {question}\n"
            f"Short answer:"
        )
        result = pipe(prompt, max_new_tokens=120, do_sample=False,
                      repetition_penalty=2.0, no_repeat_ngram_size=4)
        text = result[0].get("generated_text", "")
        if "Short answer:" in text:
            text = text.split("Short answer:")[-1].strip()
        return text if len(text) > 10 else f"Based on the video: {context[:200]}"
    except Exception as e:
        logger.warning(f"FLAN fallback failed: {e}")
    return f"Based on the video content: {context[:300]}..."
