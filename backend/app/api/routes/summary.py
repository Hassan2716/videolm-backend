"""Summary API routes."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.database import get_db, Summary
from app.models.schemas import SummaryOut
from app.services.summarization_service import SummarizationService

router = APIRouter()


@router.get("/{project_id}", response_model=list[SummaryOut])
def get_summaries(
    project_id: str,
    summary_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Summary).filter(Summary.project_id == project_id)
    if summary_type:
        q = q.filter(Summary.summary_type == summary_type)
    return q.order_by(Summary.created_at).all()


@router.post("/{project_id}/generate")
async def generate_summary(
    project_id: str,
    background_tasks: BackgroundTasks,
    summary_type: str = "medium",
    model: str = "bart",
):
    """Generate a summary on demand.
    summary_type: short | medium | detailed | bullets | academic | topic
    model: bart | t5 | pegasus | t5small
    """
    background_tasks.add_task(SummarizationService.generate, project_id, summary_type, model)
    return {"message": f"Generating {summary_type} summary with {model}…"}


@router.get("/models/availability")
def models_availability():
    """Report which summarization models can genuinely run (complete local cache).
    Models reported False will fall back to extractive TextRank summarization."""
    from app.pipeline.nlp.summarizer_v2 import model_availability
    return model_availability()


@router.get("/{project_id}/compare")
def compare_models(project_id: str, db: Session = Depends(get_db)):
    """Return all summaries grouped by model for side-by-side comparison."""
    summaries = db.query(Summary).filter(Summary.project_id == project_id).all()
    result: dict = {}
    for s in summaries:
        result.setdefault(s.model_used, []).append({
            "type": s.summary_type,
            "content": s.content,
            "word_count": s.word_count,
        })
    return result
