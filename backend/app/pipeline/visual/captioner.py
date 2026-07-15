"""
Captioner — Integrated from silent-video-segmentation project.

This is the EXACT captioner from your uploaded project, extended with:
  - caption_frame()         API for VideoLM pipeline_service.py
  - summarize_scene()       API for VideoLM pipeline_service.py
  - batch_caption_frames()  API for VideoLM pipeline_service.py

Original model: BLIP-2 (Salesforce/blip2-opt-2.7b) with BLIP fallback.
"""

from typing import Optional, List
from loguru import logger

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── Visual-type-aware prompts (from your original project) ────────────────────
VISUAL_TYPE_PROMPTS = {
    "chart":       "Question: What does this chart show? Answer:",
    "table":       "Question: What data is in this table? Answer:",
    "diagram":     "Question: What does this diagram or flowchart illustrate? Answer:",
    "graph":       "Question: What does this graph or plot show? Answer:",
    "slide":       "Question: What information is on this presentation slide? Answer:",
    "infographic": "Question: What does this infographic communicate? Answer:",
    "unknown":     "Question: Describe the visual information shown here. Answer:",
}

DEFAULT_PROMPT = "Question: Describe only the visible data visualization or diagram. Answer:"


class Captioner:
    """
    Drop-in replacement for both:
      - silent-video-segmentation  Captioner  (uses .caption() method)
      - VideoLM ImageCaptioner                (uses .caption_frame() method)
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cpu",
        max_new_tokens: int = 150,
        num_beams: int = 4,
        min_length: int = 20,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.min_length = min_length
        self.model = None
        self.processor = None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        """Try BLIP-2 first, fall back to BLIP (same logic as your original project)."""
        if not PIL_AVAILABLE:
            logger.warning("PIL not available — captioning disabled")
            return

        # ── Try BLIP-2 ────────────────────────────────────────────────────────
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            logger.info(f"Loading BLIP-2: {self.model_name}")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self.model_type = "blip2"
            logger.info("✅ BLIP-2 loaded successfully")
            return
        except Exception as e:
            logger.warning(f"BLIP-2 failed: {e}")

        # ── Fallback: BLIP ─────────────────────────────────────────────────────
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            fallback = "Salesforce/blip-image-captioning-base"
            logger.info(f"Falling back to BLIP: {fallback}")
            self.processor = BlipProcessor.from_pretrained(fallback)
            self.model = BlipForConditionalGeneration.from_pretrained(
                fallback, torch_dtype=torch.float32,
            )
            self.model_type = "blip"
            logger.info("✅ BLIP (fallback) loaded successfully")
        except Exception as e:
            logger.error(f"BLIP fallback also failed: {e} — captions unavailable")
            self.model = None

    # ── Original API (used by silent-video-segmentation job_service.py) ────────

    def caption(self, frame_path: str, visual_type: str = "unknown") -> Optional[str]:
        """Original method — kept exactly as in your project."""
        if self.model is None:
            return self._rule_based_caption(visual_type)
        try:
            import torch
            img = Image.open(frame_path).convert("RGB")
            prompt = VISUAL_TYPE_PROMPTS.get(visual_type, DEFAULT_PROMPT)

            if self.model_type == "blip2":
                inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                        min_length=self.min_length,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                    )
                caption = self.processor.decode(output[0], skip_special_tokens=True)
                if prompt in caption:
                    caption = caption.replace(prompt, "").strip()
            else:
                inputs = self.processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                    )
                caption = self.processor.decode(output[0], skip_special_tokens=True)

            return self._clean_caption(caption)

        except Exception as e:
            logger.debug(f"Captioning error for {frame_path}: {e}")
            return self._rule_based_caption(visual_type)

    # ── VideoLM API (used by videolm pipeline_service.py) ─────────────────────

    def caption_frame(self, image_path: str) -> str:
        """VideoLM API — wraps caption() with default visual_type."""
        result = self.caption(image_path, visual_type="unknown")
        return result or "Visual frame"

    def summarize_scene(self, image_paths: List[str]) -> str:
        """
        VideoLM API — summarize a scene from multiple frames.
        Captions the most representative frame (middle of scene).
        """
        if not image_paths:
            return ""
        # Use middle frame as most representative
        mid = image_paths[len(image_paths) // 2]
        return self.caption_frame(mid)

    def batch_caption_frames(self, image_paths: List[str]) -> List[str]:
        """
        VideoLM API — batch caption multiple frames.
        Processes sequentially (upgrade to batch inference if GPU available).
        """
        captions = []
        for path in image_paths:
            captions.append(self.caption_frame(path))
        return captions

    # ── Helpers (identical to your original project) ──────────────────────────

    def _clean_caption(self, text: str) -> str:
        """Remove BLIP hallucination artifacts using caption_cleaner."""
        try:
            from app.pipeline.visual.caption_cleaner import clean_caption
            return clean_caption(text)
        except Exception:
            # Inline fallback
            if not text: return ""
            import re
            text = text.strip()
            for art in ["arafed", "araffe", "araffes"]:
                text = text.replace(art, "")
            # Remove repeated words
            text = re.sub(r'\b(\w+)(\s+\1){2,}', r'\1', text)
            text = text.strip()
            if text: text = text[0].upper() + text[1:]
            return text

    def _rule_based_caption(self, visual_type: str) -> str:
        """Fallback when model is unavailable."""
        captions = {
            "chart":       "Chart or graph detected in frame.",
            "table":       "Data table detected in frame.",
            "diagram":     "Diagram or flowchart detected in frame.",
            "graph":       "Graph or plot detected in frame.",
            "slide":       "Presentation slide detected in frame.",
            "infographic": "Infographic detected in frame.",
        }
        return captions.get(visual_type, "Visual information element detected in frame.")
