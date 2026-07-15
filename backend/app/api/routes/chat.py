"""Chat with video — RAG-powered QA."""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db, ChatMessage, Project
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == req.project_id).first()
    if not project: raise HTTPException(404, "Project not found")
    if project.status != "complete": raise HTTPException(400, "Project still processing")

    # Save user message
    user_msg = ChatMessage(
        id=str(uuid.uuid4()), project_id=req.project_id,
        role="user", content=req.message,
    )
    db.add(user_msg); db.commit()

    # RAG answer
    answer, citations = await RAGService.answer(
        project_id=req.project_id,
        question=req.message,
        include_citations=req.include_citations,
    )

    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()), project_id=req.project_id,
        role="assistant", content=answer,
        citations=citations,
    )
    db.add(assistant_msg); db.commit(); db.refresh(assistant_msg)
    return assistant_msg


@router.get("/{project_id}/history", response_model=list[ChatResponse])
def get_history(project_id: str, db: Session = Depends(get_db)):
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at)
        .all()
    )


@router.delete("/{project_id}/history")
def clear_history(project_id: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.project_id == project_id).delete()
    db.commit()
    return {"message": "Cleared"}
