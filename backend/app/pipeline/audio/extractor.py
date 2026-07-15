"""Audio extraction using FFmpeg with Windows path detection."""
import os, subprocess, shutil
from loguru import logger


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


class AudioExtractor:
    @staticmethod
    def extract(video_path: str, output_dir: str) -> str:
        audio_path = os.path.join(output_dir, "audio.wav")
        if not FFMPEG:
            raise RuntimeError(
                "FFmpeg not found. Install from https://www.gyan.dev/ffmpeg/builds/ "
                "and add C:\\ffmpeg\\bin to your PATH."
            )
        cmd = [
            FFMPEG, "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-af", "loudnorm",
            audio_path, "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"FFmpeg audio error: {result.stderr[:300]}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr[:200]}")
        logger.info(f"Audio extracted -> {audio_path}")
        return audio_path
