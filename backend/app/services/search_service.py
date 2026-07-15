"""Semantic search service."""
from typing import List
from app.services.index_service import IndexService
from app.models.schemas import SearchResult


class SearchService:
    @staticmethod
    async def search(project_id: str, query: str, top_k: int = 5, search_type: str = "semantic") -> List[SearchResult]:
        results = IndexService.search(project_id, query, top_k)
        return [SearchResult(**r) for r in results]
