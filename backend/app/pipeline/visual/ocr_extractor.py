"""
OCR Extractor — from silent-video-segmentation project (exact copy).
Extracts text from visual frames using Tesseract + OpenCV preprocessing.
"""

import cv2
import numpy as np
from loguru import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available — OCR disabled")


class OCRExtractor:
    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self.config = "--oem 3 --psm 6"

    def extract(self, frame_path: str) -> str:
        """Extract and clean text from a frame image."""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return ""
            processed = self._preprocess(img)
            text = pytesseract.image_to_string(processed, lang=self.lang, config=self.config)
            return self._clean_text(text)
        except Exception as e:
            logger.debug(f"OCR error on {frame_path}: {e}")
            return ""

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w < 800:
            scale = 800 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2,
        )
        return cv2.fastNlMeansDenoising(thresh, h=10)

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = [l for l in lines if any(c.isalnum() for c in l)]
        return "\n".join(lines)
