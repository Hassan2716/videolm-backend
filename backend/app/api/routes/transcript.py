"""Transcript API routes."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db, Transcript
from app.models.schemas import TranscriptOut

router = APIRouter()


@router.get("/{project_id}", response_model=TranscriptOut)
def get_transcript(project_id: str, db: Session = Depends(get_db)):
    t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
    if not t: raise HTTPException(404, "Transcript not ready yet")
    return t


@router.get("/{project_id}/text", response_class=PlainTextResponse)
def get_transcript_text(project_id: str, db: Session = Depends(get_db)):
    t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
    if not t: raise HTTPException(404, "Transcript not ready")
    return t.full_text


@router.get("/{project_id}/srt", response_class=PlainTextResponse)
def get_transcript_srt(project_id: str, db: Session = Depends(get_db)):
    t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
    if not t: raise HTTPException(404, "Transcript not ready")

    lines = []
    for i, seg in enumerate(t.segments or [], 1):
        start = _fmt_srt(seg.get("start", 0))
        end = _fmt_srt(seg.get("end", 0))
        lines.append(f"{i}\n{start} --> {end}\n{seg.get('text','').strip()}\n")
    return "\n".join(lines)


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
