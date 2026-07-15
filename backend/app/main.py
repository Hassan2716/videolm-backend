"""VideoLM v2 — FastAPI app with all fixes."""
import os

# Load .env file BEFORE anything else — ensures GROQ_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    groq = os.environ.get("GROQ_API_KEY", "")
    if groq and groq != "your_groq_api_key_here":
        print(f"✅ GROQ_API_KEY loaded ({len(groq)} chars)")
    else:
        print("⚠️  GROQ_API_KEY not set — using FLAN-T5 fallback")
except Exception as e:
    print(f"⚠️  dotenv load failed: {e}")
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from app.core.config import settings
from app.core.database import create_tables
from app.api.routes import projects, youtube, transcript, summary, frames, study
from app.api.routes.chat_v2   import router as chat_router
from app.api.routes.search    import router as search_router
from app.api.routes.generate  import router as generate_router
from app.api.routes.presentation import router as presentation_router

# Import export — use v2 if available, fallback to original
try:
    from app.api.routes.export_v2 import router as export_router
except ImportError:
    from app.api.routes.export import router as export_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 VideoLM v2 starting…")
    create_tables()
    for d in [settings.upload_dir, settings.output_dir,
              settings.export_dir, settings.faiss_index_dir, "logs"]:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Device: {settings.device} | Whisper: {settings.whisper_model}")
    yield
    logger.info("🛑 VideoLM v2 stopping…")

app = FastAPI(title="VideoLM API v2", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.export_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
app.mount("/exports", StaticFiles(directory=settings.export_dir), name="exports")

app.include_router(projects.router,   prefix="/api/projects",   tags=["Projects"])
app.include_router(youtube.router,    prefix="/api/youtube",    tags=["YouTube"])
app.include_router(transcript.router, prefix="/api/transcript", tags=["Transcript"])
app.include_router(summary.router,    prefix="/api/summary",    tags=["Summary"])
app.include_router(frames.router,     prefix="/api/frames",     tags=["Frames"])
app.include_router(study.router,      prefix="/api/study",      tags=["Study"])
app.include_router(chat_router,       prefix="/api/chat",       tags=["Chat"])
app.include_router(export_router,     prefix="/api/export",     tags=["Export"])
app.include_router(search_router,     prefix="/api/search",     tags=["Search"])
app.include_router(generate_router,   prefix="/api/generate",   tags=["Generate"])
app.include_router(presentation_router, prefix="/api/presentation", tags=["Presentation"])

@app.get("/health")
async def health():
    return {"status":"ok","version":"2.0.0","device":settings.device}

@app.get("/")
async def root():
    return {"app":"VideoLM v2","docs":"/docs"}
