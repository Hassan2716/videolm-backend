"""Projects CRUD routes."""
import os, uuid, shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.database import get_db, Project
from app.core.config import settings
from app.models.schemas import ProjectOut
from app.services.pipeline_service import PipelineService

router = APIRouter()


@router.post("/upload", response_model=ProjectOut)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    allowed = {"mp4", "avi", "mov", "webm", "mkv"}
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: .{ext}. Allowed: {', '.join(allowed)}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(413, f"File too large: {size_mb:.0f}MB. Max: {settings.max_file_size_mb}MB")

    project_id = str(uuid.uuid4())
    video_dir = os.path.join(settings.upload_dir, project_id)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, file.filename)

    with open(video_path, "wb") as f:
        f.write(content)

    project = Project(
        id=project_id,
        source_type="local",
        source_filename=file.filename,
        video_path=video_path,
        status="pending",
        current_stage="Queued",
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    background_tasks.add_task(PipelineService.run, project_id, video_path)
    return project


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@router.get("/", response_model=list[ProjectOut])
def list_projects(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return (
        db.query(Project)
        .order_by(Project.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.delete("/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    db.delete(p)
    db.commit()
    for d in [
        os.path.join(settings.upload_dir, project_id),
        os.path.join(settings.output_dir, project_id),
        os.path.join(settings.export_dir, project_id),
    ]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    return {"message": f"Project {project_id} deleted"}
