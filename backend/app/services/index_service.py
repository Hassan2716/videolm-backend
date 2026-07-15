"""FAISS vector index — semantic search for RAG."""
import os, pickle
from typing import List, Dict, Any
from loguru import logger
import numpy as np
from app.core.config import settings


class IndexService:
    @staticmethod
    def build(project_id: str, text: str, segments: list, frames: list, chunks: list):
        embedder = _get_embedder()
        if not embedder:
            logger.warning("No embedder — skipping index build")
            return
        docs = []
        for c in chunks:
            docs.append({"text": c["text"], "source": "transcript", "start": c.get("start"), "end": c.get("end")})
        for f in frames:
            if f.get("caption"):
                docs.append({"text": f["caption"], "source": "frame", "start": f.get("ts"), "end": None})
            if f.get("ocr"):
                docs.append({"text": f["ocr"], "source": "ocr", "start": f.get("ts"), "end": None})
        if not docs:
            return
        try:
            import faiss
            embeddings = embedder.encode([d["text"] for d in docs], batch_size=32, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            idx_dir = os.path.join(settings.faiss_index_dir, project_id)
            os.makedirs(idx_dir, exist_ok=True)
            faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
            with open(os.path.join(idx_dir, "docs.pkl"), "wb") as f:
                pickle.dump(docs, f)
            logger.info(f"FAISS index: {len(docs)} docs for {project_id}")
        except Exception as e:
            logger.error(f"Index build failed: {e}")

    @staticmethod
    def search(project_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedder = _get_embedder()
        if not embedder:
            return []
        idx_dir = os.path.join(settings.faiss_index_dir, project_id)
        index_path = os.path.join(idx_dir, "index.faiss")
        docs_path = os.path.join(idx_dir, "docs.pkl")
        if not os.path.exists(index_path):
            return []
        try:
            import faiss
            index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
            q_emb = np.array(embedder.encode([query], show_progress_bar=False), dtype="float32")
            faiss.normalize_L2(q_emb)
            scores, indices = index.search(q_emb, min(top_k, len(docs)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = docs[idx]
                ts = doc.get("start")
                results.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "timestamp": _ts_label(ts) if ts is not None else None,
                    "score": float(score),
                })
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


_embedder_cache = None


def _get_embedder():
    global _embedder_cache
    if _embedder_cache:
        return _embedder_cache
    try:
        from sentence_transformers import SentenceTransformer
        _embedder_cache = SentenceTransformer(settings.embedding_model)
        return _embedder_cache
    except Exception as e:
        logger.warning(f"Embedder load failed: {e}")
        return None


def _ts_label(seconds: float) -> str:
    if seconds is None:
        return ""
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
