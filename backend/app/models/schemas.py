"""Pydantic schemas for API request/response."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class ProjectOut(BaseModel):
    id: str
    title: Optional[str] = None
    source_type: str
    source_url: Optional[str] = None
    source_filename: Optional[str] = None
    duration_seconds: Optional[float] = None
    language: str = "en"
    status: str
    progress: int
    current_stage: str
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    total_frames_scanned: int = 0
    frames_extracted: int = 0
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class YouTubeRequest(BaseModel):
    url: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptOut(BaseModel):
    id: str
    project_id: str
    full_text: str
    language: str
    word_count: int
    segments: List[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


class SummaryOut(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}

    id: str
    project_id: str
    summary_type: str
    model_used: str
    content: str
    word_count: int
    created_at: datetime


class FrameOut(BaseModel):
    id: str
    project_id: str
    frame_path: str
    timestamp_seconds: float
    timestamp_label: str
    scene_index: int
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    visual_type: Optional[str] = None

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    project_id: str
    message: str
    include_citations: bool = True


class Citation(BaseModel):
    timestamp: Optional[str] = None
    text: str
    source: str


class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = 5
    search_type: str = "semantic"


class SearchResult(BaseModel):
    text: str
    source: str
    timestamp: Optional[str] = None
    score: float


class StudyAssetOut(BaseModel):
    id: str
    project_id: str
    asset_type: str
    content: Any
    created_at: datetime

    class Config:
        from_attributes = True


class ExportRequest(BaseModel):
    project_id: str
    format: str
    content_types: List[str]


class ExportOut(BaseModel):
    id: str
    project_id: str
    format: str
    status: str
    file_path: Optional[str] = None
    download_url: Optional[str] = None

    class Config:
        from_attributes = True
