"""
Pipeline Service — Orchestrates the full multimodal AI pipeline for VideoLM.

Integrates the EXACT pipeline from silent-video-segmentation:
  Stage 1:  Video metadata (VideoProcessor)
  Stage 2:  Frame extraction (FrameExtractor — FFmpeg + OpenCV fallback)
  Stage 3:  Scene detection (SceneDetector — histogram diff)
  Stage 4:  Deduplication (Deduplicator — pHash + SSIM)
  Stage 5:  Visual detection (VisualDetector — YOLOv8 + heuristic)
  Stage 6:  OCR extraction (OCRExtractor — Tesseract + OpenCV)
  Stage 7:  Caption generation (Captioner — BLIP-2 → BLIP fallback)
  Stage 8:  NLP summarization (Summarizer — BART/T5/PEGASUS)
  Stage 9:  RAG index build (IndexService — FAISS)
  Stage 10: Save all results to database
"""

import os
import uuid
import subprocess
import time
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, Project, Transcript, Summary, Frame
from app.core.config import settings

# ── Your exact pipeline modules from silent-video-segmentation ────────────────
from app.pipeline.audio.extractor import AudioExtractor
from app.pipeline.audio.whisper_stt import WhisperSTT
from app.pipeline.video.frame_extractor import FrameExtractor
from app.pipeline.video.scene_detector import SceneDetector
from app.pipeline.video.deduplicator import Deduplicator
from app.pipeline.visual.visual_detector import VisualDetector
from app.pipeline.visual.ocr_extractor import OCRExtractor
from app.pipeline.visual.captioner import Captioner          # ← YOUR BLIP-2 model
from app.pipeline.nlp.chunker import TextChunker
from app.pipeline.nlp.summarizer_v2 import HierarchicalSummarizer
from app.services.index_service import IndexService


def _update(db: Session, pid: str, **kw):
    db.query(Project).filter(Project.id == pid).update(kw)
    db.commit()


class PipelineService:

    @staticmethod
    def run(project_id: str, video_path: str):
        """Full pipeline for a local video file."""
        db = SessionLocal()
        t0 = time.time()
        try:
            _update(db, project_id, status="processing", progress=5, current_stage="Analyzing video…")
            PipelineService._execute(db, project_id, video_path, t0)
        except Exception as e:
            logger.exception(f"Pipeline failed {project_id}: {e}")
            _update(db, project_id, status="failed", error_message=str(e), current_stage="Error")
        finally:
            db.close()

    @staticmethod
    def run_youtube(project_id: str, url: str):
        """Download YouTube video then run full pipeline."""
        db = SessionLocal()
        t0 = time.time()
        try:
            _update(db, project_id, status="processing", progress=3, current_stage="Downloading video…")
            video_path = PipelineService._download_youtube(project_id, url)
            if not video_path:
                raise RuntimeError(
                    "YouTube download failed. "
                    "Try: (1) pip install --upgrade yt-dlp  "
                    "(2) Add cookies file  "
                    "(3) Check network/firewall"
                )
            _update(db, project_id, video_path=video_path)
            PipelineService._execute(db, project_id, video_path, t0)
        except Exception as e:
            logger.exception(f"YouTube pipeline failed {project_id}: {e}")
            _update(db, project_id, status="failed", error_message=str(e), current_stage="Error")
        finally:
            db.close()

    @staticmethod
    def _download_youtube(project_id: str, url: str) -> Optional[str]:
        """yt-dlp download with multiple format fallbacks (from silent-video-segmentation)."""
        download_dir = os.path.join(settings.upload_dir, project_id)
        os.makedirs(download_dir, exist_ok=True)
        output_template = os.path.join(download_dir, "video.%(ext)s")

        format_strategies = [
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]",
            "bestvideo[height<=720]+bestaudio/best",
            "bestvideo+bestaudio/best",
            "best[ext=mp4]/best",
            "worst",
        ]

        for fmt in format_strategies:
            cmd = [
                "yt-dlp", "--format", fmt,
                "--output", output_template,
                "--no-playlist",
                "--merge-output-format", "mp4",
                "--no-check-certificates",
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "--extractor-retries", "3",
                "--fragment-retries", "3",
                "--retry-sleep", "5",
                "--socket-timeout", "30",
                "--geo-bypass",
            ]
            # Use cookies file if it exists
            if os.path.exists(settings.youtube_cookies_file):
                cmd += ["--cookies", settings.youtube_cookies_file]
            cmd.append(url)

            logger.info(f"yt-dlp trying format: {fmt[:80]}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.stdout:
                    logger.info(f"yt-dlp: {result.stdout[-300:]}")
                if result.returncode != 0:
                    logger.warning(f"yt-dlp stderr: {result.stderr[-300:]}")
                    continue
                for fname in os.listdir(download_dir):
                    if fname.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi")):
                        full_path = os.path.join(download_dir, fname)
                        size_mb = os.path.getsize(full_path) / 1024 / 1024
                        logger.info(f"Downloaded: {fname} ({size_mb:.1f} MB)")
                        return full_path
            except subprocess.TimeoutExpired:
                logger.error("yt-dlp timed out (600s)")
                continue
            except FileNotFoundError:
                raise RuntimeError("yt-dlp not found — run: pip install yt-dlp")

        return None

    @staticmethod
    def _execute(db: Session, project_id: str, video_path: str, t0: float):
        """Execute all pipeline stages."""
        output_dir = os.path.join(settings.output_dir, project_id)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting pipeline for {project_id}")

        # ── Stage 1: Audio extraction ──────────────────────────────────────────
        _update(db, project_id, progress=10, current_stage="Extracting audio…")
        audio_path = AudioExtractor.extract(video_path, output_dir)

        # ── Stage 2: Whisper transcription ─────────────────────────────────────
        _update(db, project_id, progress=18, current_stage="Transcribing with Whisper…")
        stt = WhisperSTT(model=settings.whisper_model, device=settings.device)
        stt_result = stt.transcribe(audio_path)

        transcript = Transcript(
            id=str(uuid.uuid4()), project_id=project_id,
            full_text=stt_result["text"],
            language=stt_result.get("language", "en"),
            word_count=len(stt_result["text"].split()),
            segments=stt_result.get("segments", []),
        )
        db.add(transcript); db.commit()
        logger.info(f"Transcript: {transcript.word_count} words, lang={transcript.language}")

        # ── Stage 3: Frame extraction ──────────────────────────────────────────
        _update(db, project_id, progress=28, current_stage="Extracting keyframes…")
        extractor = FrameExtractor(video_path, output_dir)
        raw_frames = extractor.extract_keyframes()
        logger.info(f"Extracted {len(raw_frames)} raw frames")

        # ── Stage 4: Scene detection ───────────────────────────────────────────
        _update(db, project_id, progress=36, current_stage="Detecting scene changes…")
        scene_detector = SceneDetector(threshold=settings.scene_threshold)
        scene_frames = scene_detector.filter_scene_changes(raw_frames)
        logger.info(f"After scene detection: {len(scene_frames)} frames")

        # ── Stage 5: Deduplication ─────────────────────────────────────────────
        _update(db, project_id, progress=42, current_stage="Removing duplicate frames…")
        dedup = Deduplicator(
            similarity_threshold=settings.similarity_threshold,
            hash_threshold=settings.perceptual_hash_threshold,
        )
        unique_frames = dedup.deduplicate(scene_frames)
        logger.info(f"After dedup: {len(unique_frames)} frames")
        _update(db, project_id, total_frames_scanned=len(raw_frames))

        # ── Stage 6: Visual detection (YOLOv8 + heuristic) ────────────────────
        _update(db, project_id, progress=52, current_stage="Detecting visual elements (YOLOv8)…")
        detector = VisualDetector(
            confidence_threshold=settings.confidence_threshold,
            device=settings.device,
        )
        detected_frames = detector.detect(unique_frames)
        logger.info(f"Detected informative visuals: {len(detected_frames)} frames")

        # ── Stage 7: OCR extraction ────────────────────────────────────────────
        _update(db, project_id, progress=62, current_stage="Extracting text with OCR…")
        ocr = OCRExtractor(lang=settings.ocr_lang)
        for frame in detected_frames:
            frame["ocr_text"] = ocr.extract(frame["frame_path"])

        # ── Stage 8: BLIP-2 Caption generation ────────────────────────────────
        _update(db, project_id, progress=72, current_stage="Generating AI captions (BLIP-2)…")
        captioner = Captioner(
            model_name=settings.blip2_model,
            device=settings.device,
            max_new_tokens=settings.blip2_max_new_tokens,
        )
        frame_summaries = []
        for i, frame in enumerate(detected_frames):
            # Use .caption() — the ORIGINAL method from your project
            frame["caption"] = captioner.caption(
                frame["frame_path"],
                visual_type=frame.get("visual_type", "unknown"),
            )
            frame_summaries.append({
                "caption": frame["caption"],
                "ocr": frame.get("ocr_text", ""),
                "ts": frame["timestamp_seconds"],
            })
            pct = 72 + int((i / max(len(detected_frames), 1)) * 12)
            _update(db, project_id,
                    current_stage=f"Captioning frame {i+1}/{len(detected_frames)}…",
                    progress=pct)

        # ── Stage 9: NLP + summarization ──────────────────────────────────────
        _update(db, project_id, progress=86, current_stage="Generating summaries…")
        chunker = TextChunker()
        chunks = chunker.chunk(transcript.full_text, transcript.segments)

        summarizer = HierarchicalSummarizer(device=settings.device)
        for stype, model_key in [
            ("short", "bart"), ("medium", "bart"),
            ("bullets", "bart"), ("academic", "t5"),
        ]:
            try:
                content = summarizer.summarize(
                    transcript.full_text, summary_type=stype,
                    model_key=model_key, segments=transcript.segments)
                s = Summary(
                    id=str(uuid.uuid4()), project_id=project_id,
                    summary_type=stype, model_used=model_key,
                    content=content, word_count=len(content.split()),
                )
                db.add(s)
            except Exception as e:
                logger.warning(f"Summary {stype} failed: {e}")
        db.commit()

        # ── Stage 10: Save frames to database ─────────────────────────────────
        _update(db, project_id, progress=92, current_stage="Saving results…")
        saved_count = 0
        for frame in detected_frames:
            db_frame = Frame(
                id=str(uuid.uuid4()), project_id=project_id,
                frame_path=os.path.relpath(frame["frame_path"], settings.output_dir),
                timestamp_seconds=frame["timestamp_seconds"],
                timestamp_label=_ts_label(frame["timestamp_seconds"]),
                scene_index=frame.get("frame_index", 0),
                caption=frame.get("caption"),
                ocr_text=frame.get("ocr_text"),
                visual_type=frame.get("visual_type", "unknown"),
            )
            db.add(db_frame)
            saved_count += 1
        db.commit()

        # ── Stage 11: Build FAISS RAG index ───────────────────────────────────
        _update(db, project_id, progress=96, current_stage="Building search index…")
        # Build TF-IDF index (always works — no sentence_transformers needed)
        try:
            from app.services.rag_v2 import build_index as build_tfidf
            build_tfidf(project_id=project_id, full_text=transcript.full_text,
                        segments=transcript.segments, frames=frame_summaries, chunks=chunks)
            logger.info(f"TF-IDF index built for {project_id}")
        except Exception as e:
            logger.warning(f"TF-IDF index failed: {e}")

        # Optional FAISS index
        IndexService.build(
            project_id=project_id,
            text=transcript.full_text,
            segments=transcript.segments,
            frames=frame_summaries,
            chunks=chunks,
        )

        # ── Complete ───────────────────────────────────────────────────────────
        elapsed = round(time.time() - t0, 1)
        _update(db, project_id,
                status="complete", progress=100,
                current_stage="Complete",
                frames_extracted=saved_count,
                processing_time_seconds=elapsed,
                completed_at=datetime.utcnow())
        logger.info(f"✅ Pipeline complete for {project_id} — {saved_count} visuals in {elapsed}s")


def _ts_label(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
