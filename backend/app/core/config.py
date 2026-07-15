"""Application configuration — merged from both projects."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "VideoLM"
    debug: bool = False
    secret_key: str = "change-me"

    # Storage
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    export_dir: str = "./exports"
    max_file_size_mb: int = 2000

    # Database
    database_url: str = "sqlite:///./videolm.db"

    # ── Settings from silent-video-segmentation ──────────────────────────────
    # Frame extraction
    frame_sample_rate: int = 1           # fps for interval extraction
    scene_threshold: float = 0.35        # scene change sensitivity
    similarity_threshold: float = 0.92  # SSIM dedup threshold
    perceptual_hash_threshold: int = 10  # pHash Hamming distance
    confidence_threshold: float = 0.65  # visual detection confidence

    # BLIP-2 (your captioning model)
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    blip2_max_new_tokens: int = 150
    blip2_num_beams: int = 4
    blip2_min_length: int = 20

    # OCR
    ocr_lang: str = "eng"

    # ── VideoLM AI settings ───────────────────────────────────────────────────
    device: str = "cpu"
    whisper_model: str = "base"
    use_faster_whisper: bool = False
    summarization_model: str = "facebook/bart-large-cnn"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qa_model: str = "google/flan-t5-base"

    # Vector DB
    vector_db: str = "faiss"
    faiss_index_dir: str = "./faiss_indexes"

    # YouTube
    youtube_cookies_file: str = "./youtube.com_cookies.txt"

    # TTS

    # AI APIs
    groq_api_key: str = ""  # Get free key at https://console.groq.com

    # CORS
    backend_cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


settings = Settings()
