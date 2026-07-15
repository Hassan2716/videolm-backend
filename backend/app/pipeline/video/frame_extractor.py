"""
Frame Extractor — from silent-video-segmentation project.
Uses FFmpeg to extract keyframes + interval sampling.
Falls back to OpenCV if FFmpeg is not found on Windows.
"""

import os
import subprocess
import shutil
from typing import List, Dict, Any

import cv2
from loguru import logger

from app.core.config import settings


def _find_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    for p in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.isfile(p):
            return p
    return None


FFMPEG = _find_ffmpeg()


class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames_raw")
        os.makedirs(self.frames_dir, exist_ok=True)

    def extract(self) -> List[Dict[str, Any]]:
        """Alias used by VideoLM pipeline service."""
        return self.extract_keyframes()

    def extract_keyframes(self) -> List[Dict[str, Any]]:
        """Two-pass extraction: I-frames + 1fps interval sampling."""
        if FFMPEG:
            logger.info(f"Extracting frames with FFmpeg: {FFMPEG}")
            iframe_frames = self._extract_iframes()
            sample_frames = self._extract_interval(fps=settings.frame_sample_rate)
            return self._merge_frames(iframe_frames, sample_frames)
        else:
            logger.warning("FFmpeg not found — using OpenCV extraction")
            return self._extract_opencv()

    def _extract_iframes(self) -> List[Dict[str, Any]]:
        iframe_dir = os.path.join(self.frames_dir, "iframes")
        os.makedirs(iframe_dir, exist_ok=True)
        cmd = [
            FFMPEG, "-i", self.video_path,
            "-vf", "select=eq(pict_type\\,I)",
            "-fps_mode", "vfr",
            "-q:v", "2",
            os.path.join(iframe_dir, "frame_%06d.jpg"),
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, timeout=300, check=False, capture_output=True)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg I-frame extraction timed out")
        except Exception as e:
            logger.error(f"FFmpeg iframe error: {e}")
        return self._frames_from_dir(iframe_dir, "iframe")

    def _extract_interval(self, fps: int = 1) -> List[Dict[str, Any]]:
        interval_dir = os.path.join(self.frames_dir, "interval")
        os.makedirs(interval_dir, exist_ok=True)
        cmd = [
            FFMPEG, "-i", self.video_path,
            "-vf", f"fps={fps},scale=1280:-1",
            "-q:v", "3",
            os.path.join(interval_dir, "frame_%06d.jpg"),
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, timeout=600, check=False, capture_output=True)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg interval extraction timed out")
        except Exception as e:
            logger.error(f"FFmpeg interval error: {e}")
        frames = self._frames_from_dir(interval_dir, "interval")
        for i, frame in enumerate(frames):
            frame["timestamp_seconds"] = i / fps
        return frames

    def _extract_opencv(self) -> List[Dict[str, Any]]:
        """OpenCV fallback — no FFmpeg required."""
        opencv_dir = os.path.join(self.frames_dir, "opencv")
        os.makedirs(opencv_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"OpenCV cannot open: {self.video_path}")
            return []
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_every = max(1, int(fps_video))
        frames, idx, saved = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                path = os.path.join(opencv_dir, f"frame_{saved:06d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                frames.append({
                    "frame_path": path,
                    "frame_index": saved,
                    "timestamp_seconds": idx / fps_video,
                    "source": "opencv",
                })
                saved += 1
            idx += 1
        cap.release()
        logger.info(f"OpenCV extracted {len(frames)} frames")
        return frames

    def _frames_from_dir(self, directory: str, source: str) -> List[Dict[str, Any]]:
        if not os.path.exists(directory):
            return []
        files = sorted(f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return [
            {
                "frame_path": os.path.join(directory, fname),
                "frame_index": i,
                "timestamp_seconds": float(i),
                "source": source,
            }
            for i, fname in enumerate(files)
            if os.path.getsize(os.path.join(directory, fname)) > 0
        ]

    def _merge_frames(self, iframes: List[Dict], sample_frames: List[Dict]) -> List[Dict[str, Any]]:
        merged = list(sample_frames)
        sample_timestamps = {f["timestamp_seconds"] for f in sample_frames}
        for iframe in iframes:
            ts = iframe["timestamp_seconds"]
            if not any(abs(ts - s) < 0.5 for s in sample_timestamps):
                merged.append(iframe)
        merged.sort(key=lambda x: x["timestamp_seconds"])
        return merged
