"""Frames route — returns frames with cleaned captions and OCR."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db, Frame
from app.models.schemas import FrameOut
from loguru import logger
from typing import List

router = APIRouter()


@router.get("/{project_id}", response_model=List[FrameOut])
def get_frames(project_id: str, db: Session = Depends(get_db)):
    frames = (
        db.query(Frame)
        .filter(Frame.project_id == project_id)
        .order_by(Frame.timestamp_seconds)
        .all()
    )

    # Build clean response dicts without touching ORM objects
    result = []
    for f in frames:
        try:
            from app.pipeline.visual.caption_cleaner import clean_frame_data
            cap, ocr = clean_frame_data(f.caption or "", f.ocr_text or "")
        except Exception as e:
            logger.warning(f"Caption clean failed: {e}")
            cap = f.caption or ""
            ocr = f.ocr_text or ""

        result.append(FrameOut(
            id=f.id,
            project_id=f.project_id,
            frame_path=f.frame_path,
            timestamp_seconds=f.timestamp_seconds,
            timestamp_label=f.timestamp_label,
            scene_index=f.scene_index,
            caption=cap,
            ocr_text=ocr,
            visual_type=f.visual_type,
        ))
    return result
