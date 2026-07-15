"""Export center route."""
import uuid
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.core.database import get_db, ExportJob
from app.models.schemas import ExportRequest, ExportOut
from app.services.export_service import ExportService
from app.core.config import settings
import os

router = APIRouter()

@router.post("/", response_model=ExportOut)
async def create_export(req: ExportRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job = ExportJob(
        id=str(uuid.uuid4()), project_id=req.project_id,
        format=req.format, content_types=req.content_types, status="pending",
    )
    db.add(job); db.commit(); db.refresh(job)
    background_tasks.add_task(ExportService.generate, job.id, req.project_id, req.format, req.content_types)
    return ExportOut(id=job.id, project_id=job.project_id, format=job.format, status=job.status, file_path=None, download_url=None)

@router.get("/{export_id}/status", response_model=ExportOut)
def get_status(export_id: str, db: Session = Depends(get_db)):
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job: raise HTTPException(404, "Export not found")
    url = f"/exports/{job.project_id}/{os.path.basename(job.file_path)}" if job.file_path else None
    return ExportOut(id=job.id, project_id=job.project_id, format=job.format, status=job.status, file_path=job.file_path, download_url=url)

@router.get("/{export_id}/download")
def download(export_id: str, db: Session = Depends(get_db)):
    job = db.query(ExportJob).filter(ExportJob.id == export_id).first()
    if not job or not job.file_path: raise HTTPException(404, "File not ready")
    return FileResponse(job.file_path, filename=os.path.basename(job.file_path))
