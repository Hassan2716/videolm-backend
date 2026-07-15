"""Search route — TF-IDF retrieval."""
from typing import List
from fastapi import APIRouter
from app.models.schemas import SearchRequest, SearchResult

router = APIRouter()

@router.post("/", response_model=List[SearchResult])
async def search(req: SearchRequest):
    try:
        from app.services.rag_v2 import retrieve
        results = retrieve(req.project_id, req.query, req.top_k)
        return [SearchResult(text=r["text"],source=r.get("source","transcript"),
                             timestamp=r.get("timestamp"),
                             score=round(r.get("score",0.0),4)) for r in results]
    except Exception as e:
        from loguru import logger; logger.error(f"Search failed: {e}"); return []
