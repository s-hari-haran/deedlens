"""
DeedLens Configuration
Centralized configuration using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = "DeedLens"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Paths - use factory defaults to compute at instantiation time
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    data_dir: Optional[Path] = Field(default=None)
    index_dir: Optional[Path] = Field(default=None)
    
    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/deedlens.db"),
        description="PostgreSQL or SQLite connection string"
    )
    
    # OCR
    ocr_backend: str = Field(default="tesseract", description="tesseract, easyocr, or groq")
    ocr_languages: List[str] = Field(default=["en"])
    ocr_dpi: int = 200
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Search
    search_default_k: int = 10
    search_semantic_weight: float = 0.7
    search_keyword_weight: float = 0.3
    
    # FAISS
    faiss_index_type: str = "flat"  # flat, cosine, ivf
    
    # API Security
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    cors_origins: List[str] = Field(default=["http://localhost:8501", "http://localhost:3000"])
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    def model_post_init(self, __context) -> None:
        """Set derived paths after initialization."""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.index_dir is None:
            self.index_dir = self.data_dir / "index"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_path(self) -> Path:
        """Get the SQLite database file path."""
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            if db_path.startswith("./"):
                return self.base_dir / db_path[2:]
            return Path(db_path)
        return self.data_dir / "deedlens.db"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Convenience access
settings = get_settings()


if __name__ == "__main__":
    # Print current settings
    s = get_settings()
    print(f"App: {s.app_name} v{s.app_version}")
    print(f"Base Dir: {s.base_dir}")
    print(f"Data Dir: {s.data_dir}")
    print(f"Database: {s.database_path}")
    print(f"OCR Backend: {s.ocr_backend}")
    print(f"Embedding Model: {s.embedding_model}")
    print(f"Log Level: {s.log_level}")
