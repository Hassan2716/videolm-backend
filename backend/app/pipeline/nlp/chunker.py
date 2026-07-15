"""Text chunking for RAG and keyphrase extraction."""
from typing import List, Dict, Any


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, segments: List[Dict] = None) -> List[Dict[str, Any]]:
        if segments:
            return self._segment_chunks(segments)
        return self._word_chunks(text)

    def _segment_chunks(self, segments):
        chunks, current, current_ts = [], [], None
        for seg in segments:
            words = seg.get("text", "").split()
            if not current_ts:
                current_ts = seg.get("start", 0)
            current.extend(words)
            if len(current) >= self.chunk_size:
                chunks.append({"text": " ".join(current), "start": current_ts, "end": seg.get("end", 0)})
                current = current[-self.overlap:]
                current_ts = seg.get("start", 0)
        if current:
            chunks.append({"text": " ".join(current), "start": current_ts, "end": None})
        return chunks

    def _word_chunks(self, text):
        words = text.split()
        return [
            {"text": " ".join(words[i:i + self.chunk_size]), "start": None, "end": None}
            for i in range(0, len(words), self.chunk_size - self.overlap)
        ]


class KeyphraseExtractor:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model:
            return
        try:
            from keybert import KeyBERT
            self._model = KeyBERT()
        except Exception:
            self._model = None

    def extract(self, text: str, top_n: int = 15) -> List[str]:
        self._load()
        if self._model:
            try:
                kws = self._model.extract_keywords(text[:5000], top_n=top_n, stop_words="english")
                return [kw[0] for kw in kws]
            except Exception:
                pass
        import re
        from collections import Counter
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stops = {"this", "that", "with", "from", "they", "have", "were", "been", "their", "also"}
        return [w for w, _ in Counter(w for w in words if w not in stops).most_common(top_n)]
