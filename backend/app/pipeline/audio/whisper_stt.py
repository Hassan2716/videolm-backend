"""Whisper STT — openai-whisper with faster-whisper fallback."""
from typing import Dict, Any
from loguru import logger


class WhisperSTT:
    def __init__(self, model: str = "base", device: str = "cpu"):
        self.model_size = model
        self.device = device
        self._model = None
        self._backend = None

    def _load(self):
        if self._model:
            return
        # Try openai-whisper
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size}…")
            self._model = whisper.load_model(self.model_size, device=self.device)
            self._backend = "whisper"
            logger.info("✅ Whisper loaded")
            return
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")
        # Fallback: faster-whisper
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
            self._backend = "faster_whisper"
            logger.info("✅ faster-whisper loaded")
        except Exception as e:
            raise RuntimeError(f"No Whisper backend available: {e}")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        self._load()
        if self._backend == "whisper":
            return self._openai(audio_path)
        return self._faster(audio_path)

    def _openai(self, audio_path: str) -> Dict:
        # word_timestamps=False: the word-alignment path in openai-whisper crashes on
        # PyTorch 2.x (SDPA returns qk=None -> 'NoneType' is not subscriptable). We only
        # use segment-level start/end, which are produced regardless of this flag.
        result = self._model.transcribe(audio_path, word_timestamps=False, verbose=False)
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip(), "speaker": None}
            for s in result.get("segments", [])
        ]
        return {"text": result["text"].strip(), "language": result.get("language", "en"), "segments": segments}

    def _faster(self, audio_path: str) -> Dict:
        segs_raw, info = self._model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        full, segments = [], []
        for seg in segs_raw:
            full.append(seg.text)
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(), "speaker": None})
        return {"text": " ".join(full).strip(), "language": info.language, "segments": segments}
