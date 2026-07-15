"""Presentation generation API route."""
import uuid, os
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.core.database import get_db, Project, Transcript, Summary, Frame
from app.core.config import settings

router = APIRouter()

class PresentationReq(BaseModel):
    num_slides: int = 10
    theme: str = "modern"
    include_images: bool = True
    summary_type: str = "medium"

_jobs: dict = {}

@router.post("/{project_id}/generate")
async def generate_presentation(
    project_id: str,
    req: PresentationReq,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p: raise HTTPException(404, "Project not found")
    if p.status != "complete": raise HTTPException(400, "Project not ready")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status":"pending","progress":0,"message":"Queued","result":None,"error":None}
    background_tasks.add_task(_run, job_id, project_id, req)
    return {"job_id": job_id}

@router.get("/status/{job_id}")
def status(job_id: str):
    job = _jobs.get(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return job

@router.get("/{project_id}/download/{filename}")
def download(project_id: str, filename: str):
    path = os.path.join(settings.export_dir, project_id, filename)
    if not os.path.exists(path): raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

def _update(jid, **kw): _jobs[jid].update(kw)

def _run(job_id: str, project_id: str, req: PresentationReq):
    try:
        _update(job_id, status="running", progress=10, message="Loading content…")
        from app.core.database import SessionLocal
        db = SessionLocal()
        t = db.query(Transcript).filter(Transcript.project_id == project_id).first()
        s_list = db.query(Summary).filter(
            Summary.project_id == project_id,
            Summary.summary_type == req.summary_type
        ).first()
        if not s_list:
            s_list = db.query(Summary).filter(Summary.project_id == project_id).first()
        frames_raw = db.query(Frame).filter(Frame.project_id == project_id)\
                       .order_by(Frame.timestamp_seconds).all()
        db.close()

        transcript_text = t.full_text if t else ""
        summary_text = s_list.content if s_list else transcript_text[:3000]

        frames = [{"frame_path": f.frame_path, "caption": f.caption,
                   "ocr_text": f.ocr_text, "ts": f.timestamp_seconds}
                  for f in frames_raw if f.caption]

        _update(job_id, progress=30, message=f"Planning {req.num_slides} slides…")

        from app.services.presentation_service import generate_pptx
        path = generate_pptx(
            project_id=project_id,
            transcript=transcript_text,
            summary=summary_text,
            frames=frames,
            num_slides=req.num_slides,
            theme_name=req.theme,
            include_images=req.include_images,
        )

        _update(job_id, status="complete", progress=100,
                message="Presentation ready!",
                result={"path": path, "filename": os.path.basename(path)})
    except Exception as e:
        from loguru import logger
        logger.exception(f"Presentation job failed: {e}")
        _update(job_id, status="failed", error=str(e))
