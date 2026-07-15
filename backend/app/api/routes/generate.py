"""On-demand generation API routes."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.core.database import get_db, Project
from app.services.generation_service import (
    create_generation_job, get_job_status,
    generate_summary_async, generate_quiz_async,
    generate_flashcards_async, generate_mindmap_async,
)

router = APIRouter()

class SummaryReq(BaseModel):
    model_config = {"protected_namespaces": ()}

    summary_type: str = "medium"
    model_key: str = "bart"

class QuizReq(BaseModel):
    num_questions: int = 10
    difficulty: str = "medium"
    question_types: List[str] = ["mcq","true_false","fill_blank"]

class FlashReq(BaseModel):
    num_cards: int = 20

def _check(project_id, db):
    p = db.query(Project).filter(Project.id==project_id).first()
    if not p: raise HTTPException(404,"Project not found")
    if p.status not in ("complete","processing"): raise HTTPException(400,"Project not ready")

@router.post("/{project_id}/summary")
async def start_summary(project_id, req: SummaryReq, bg: BackgroundTasks, db: Session=Depends(get_db)):
    _check(project_id, db)
    jid = create_generation_job(project_id, "summary", req.dict())
    bg.add_task(generate_summary_async, jid, project_id, req.summary_type, req.model_key)
    return {"job_id":jid,"status":"pending"}

@router.post("/{project_id}/quiz")
async def start_quiz(project_id, req: QuizReq, bg: BackgroundTasks, db: Session=Depends(get_db)):
    _check(project_id, db)
    jid = create_generation_job(project_id, "quiz", req.dict())
    bg.add_task(generate_quiz_async, jid, project_id, req.dict())
    return {"job_id":jid,"status":"pending"}

@router.post("/{project_id}/flashcards")
async def start_flashcards(project_id, req: FlashReq, bg: BackgroundTasks, db: Session=Depends(get_db)):
    _check(project_id, db)
    jid = create_generation_job(project_id, "flashcard", req.dict())
    bg.add_task(generate_flashcards_async, jid, project_id, req.dict())
    return {"job_id":jid,"status":"pending"}

@router.post("/{project_id}/mindmap")
async def start_mindmap(project_id, bg: BackgroundTasks, db: Session=Depends(get_db)):
    _check(project_id, db)
    jid = create_generation_job(project_id, "mindmap", {})
    bg.add_task(generate_mindmap_async, jid, project_id, {})
    return {"job_id":jid,"status":"pending"}

@router.get("/status/{job_id}")
def poll(job_id):
    job = get_job_status(job_id)
    if not job: raise HTTPException(404,"Job not found")
    return job
