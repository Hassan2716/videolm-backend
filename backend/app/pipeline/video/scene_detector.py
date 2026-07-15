"""
Scene Detector — from silent-video-segmentation project.
Filters frames to only keep significant scene changes.
Uses histogram comparison + frame differencing.
"""

from typing import List, Dict, Any
import cv2
import numpy as np
from loguru import logger


class SceneDetector:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def filter(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Alias for filter_scene_changes — used by VideoLM pipeline."""
        return self.filter_scene_changes(frames)

    def filter_scene_changes(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only frames that represent significant scene changes."""
        if len(frames) < 2:
            return frames

        logger.info(f"Scene detection on {len(frames)} frames (threshold={self.threshold})")
        kept_frames = [frames[0]]
        prev_hist = self._compute_histogram(frames[0]["frame_path"])

        for frame in frames[1:]:
            curr_hist = self._compute_histogram(frame["frame_path"])
            if curr_hist is None:
                continue
            diff = self._histogram_diff(prev_hist, curr_hist)
            if diff > self.threshold:
                kept_frames.append(frame)
                prev_hist = curr_hist
            else:
                last_ts = kept_frames[-1]["timestamp_seconds"]
                curr_ts = frame["timestamp_seconds"]
                if curr_ts - last_ts > 10.0:
                    kept_frames.append(frame)
                    prev_hist = curr_hist

        logger.info(f"Scene detection: {len(frames)} -> {len(kept_frames)} frames")
        return kept_frames

    def _compute_histogram(self, frame_path: str):
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return None
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None

    def _histogram_diff(self, h1, h2) -> float:
        try:
            return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        except Exception:
            return 0.0
