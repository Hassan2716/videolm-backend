"""
Deduplicator — exact copy from silent-video-segmentation.
pHash + SSIM duplicate frame removal.
"""
from typing import List, Dict, Any
import cv2
import numpy as np
from loguru import logger

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not available — using SSIM-only deduplication")


class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.92, hash_threshold: int = 10):
        self.similarity_threshold = similarity_threshold
        self.hash_threshold = hash_threshold

    def deduplicate(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(frames) < 2:
            return frames
        logger.info(f"Deduplicating {len(frames)} frames")
        unique = self._phash_dedup(frames) if IMAGEHASH_AVAILABLE else self._ssim_dedup(frames)
        logger.info(f"After dedup: {len(frames)} -> {len(unique)} frames")
        return unique

    def _phash_dedup(self, frames):
        unique_frames, seen_hashes = [], []
        for frame in frames:
            try:
                img = Image.open(frame["frame_path"])
                h = imagehash.phash(img)
                if not any(abs(h - sh) < self.hash_threshold for sh in seen_hashes):
                    unique_frames.append(frame)
                    seen_hashes.append(h)
            except Exception:
                unique_frames.append(frame)
        return unique_frames

    def _ssim_dedup(self, frames):
        unique_frames = [frames[0]]
        prev_gray = self._to_gray(frames[0]["frame_path"])
        for frame in frames[1:]:
            curr_gray = self._to_gray(frame["frame_path"])
            if curr_gray is None:
                continue
            if self._ssim(prev_gray, curr_gray) < self.similarity_threshold:
                unique_frames.append(frame)
                prev_gray = curr_gray
        return unique_frames

    def _to_gray(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            return cv2.cvtColor(cv2.resize(img, (256, 144)), cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    def _ssim(self, img1, img2) -> float:
        try:
            from skimage.metrics import structural_similarity
            score, _ = structural_similarity(img1, img2, full=True)
            return float(score)
        except Exception:
            diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
            return 1.0 - (diff / 255.0)
