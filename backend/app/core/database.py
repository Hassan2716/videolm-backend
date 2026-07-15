"""Database models and session management."""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Text, JSON, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True)
    source_type = Column(String)           # youtube | local
    source_url = Column(String, nullable=True)
    source_filename = Column(String, nullable=True)
    video_path = Column(String, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    language = Column(String, default="en")
    status = Column(String, default="pending")   # pending|processing|complete|failed
    progress = Column(Integer, default=0)
    current_stage = Column(String, default="")
    error_message = Column(Text, nullable=True)
    video_metadata = Column(JSON, nullable=True)  # renamed — "metadata" is reserved by SQLAlchemy
    processing_time_seconds = Column(Float, nullable=True)
    total_frames_scanned = Column(Integer, default=0)
    frames_extracted = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    full_text = Column(Text)
    language = Column(String, default="en")
    word_count = Column(Integer, default=0)
    segments = Column(JSON)        # [{start, end, text, speaker}]
    created_at = Column(DateTime, default=datetime.utcnow)


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    summary_type = Column(String)   # short|medium|detailed|bullets|academic|topic
    model_used = Column(String)
    content = Column(Text)
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Frame(Base):
    __tablename__ = "frames"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    frame_path = Column(String)
    timestamp_seconds = Column(Float)
    timestamp_label = Column(String)
    scene_index = Column(Integer, default=0)
    caption = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    visual_type = Column(String, nullable=True)
    embedding_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    role = Column(String)       # user | assistant
    content = Column(Text)
    citations = Column(JSON, nullable=True)   # [{timestamp, text, source}]
    created_at = Column(DateTime, default=datetime.utcnow)


class StudyAsset(Base):
    __tablename__ = "study_assets"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    asset_type = Column(String)   # flashcard|quiz|mindmap|notes
    content = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(String, primary_key=True)
    project_id = Column(String, index=True)
    format = Column(String)
    content_types = Column(JSON)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
