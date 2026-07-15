"""Summarization pipeline — BART, T5, PEGASUS, TextRank fallback."""
from loguru import logger

MODELS = {
    "bart":    "facebook/bart-large-cnn",
    "t5":      "t5-base",
    "pegasus": "google/pegasus-xsum",
    "t5small": "t5-small",
}
PROMPTS = {
    "short":    "Summarize in 2-3 sentences: ",
    "medium":   "Provide a comprehensive summary: ",
    "detailed": "Provide a detailed summary with key points: ",
    "bullets":  "Summarize as bullet points: ",
    "academic": "Write an academic summary with key concepts and findings: ",
    "topic":    "Identify and summarize the main topics: ",
}
MAX_LENGTHS = {"short": 80, "medium": 200, "detailed": 400, "bullets": 250, "academic": 350, "topic": 300}


class Summarizer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._models = {}

    def _load(self, model_key: str):
        if model_key in self._models:
            return self._models[model_key]
        model_name = MODELS.get(model_key, MODELS["bart"])
        try:
            from transformers import pipeline
            logger.info(f"Loading summarizer: {model_name}")
            pipe = pipeline(
                "summarization", model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype="auto",
            )
            self._models[model_key] = pipe
            logger.info(f"✅ {model_key} loaded")
            return pipe
        except Exception as e:
            logger.warning(f"Model {model_key} failed: {e}")
            return None

    def summarize(self, text: str, summary_type: str = "medium", model_key: str = "bart") -> str:
        words = text.split()
        if len(words) > 900:
            text = " ".join(words[:900])
        pipe = self._load(model_key)
        if pipe:
            try:
                prefix = PROMPTS.get(summary_type, "")
                input_text = prefix + text if model_key.startswith("t5") else text
                max_len = MAX_LENGTHS.get(summary_type, 200)
                result = pipe(input_text, max_length=max_len, min_length=30, do_sample=False)
                content = result[0]["summary_text"]
                if summary_type == "bullets":
                    content = "\n".join(f"• {s.strip()}" for s in content.replace(". ", ".\n").split("\n") if s.strip())
                return content
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        return self._textrank(text, summary_type)

    def _textrank(self, text, summary_type):
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            n = {"short": 3, "medium": 6, "detailed": 10}.get(summary_type, 5)
            return " ".join(str(s) for s in TextRankSummarizer()(parser.document, n))
        except Exception:
            return text[:500] + "..."
