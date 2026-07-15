"""Study assets route — flashcards, quiz, mind map."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db, StudyAsset
from app.models.schemas import StudyAssetOut
from app.services.study_service import StudyService

router = APIRouter()


@router.get("/{project_id}", response_model=list[StudyAssetOut])
def get_assets(
    project_id: str,
    asset_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(StudyAsset).filter(StudyAsset.project_id == project_id)
    if asset_type:
        q = q.filter(StudyAsset.asset_type == asset_type)
    return q.all()


@router.post("/{project_id}/generate")
async def generate_assets(
    project_id: str,
    background_tasks: BackgroundTasks,
    asset_type: str = "flashcard",
):
    """
    Generate study assets.
    asset_type: flashcard | quiz | mindmap | notes
    """
    background_tasks.add_task(StudyService.generate, project_id, asset_type)
    return {"message": f"Generating {asset_type} for project {project_id}…"}
