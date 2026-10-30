"""Configuration module for FinRAG-Eval.

Loads environment variables, manages directory paths, model constants,
chunking parameters, and automatically ensures required directories exist.
"""

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file explicitly if present
load_dotenv()

# Root directory of the repository (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings and runtime constants."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Base Paths
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_pdfs_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw_pdfs")
    processed_md_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed_md")
    chroma_db_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")
    bm25_index_dir: Path = Field(default=PROJECT_ROOT / "data" / "bm25_index")

    # Local CPU Model Constants (Source: Techstack.md Section 3)
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    reranker_model: str = Field(default="ms-marco-MiniLM-L-12-v2")

    # LLM Inference Models (Source: Techstack.md Section 4 & Groq Platform)
    main_agent_model: str = Field(
        default="openai/gpt-oss-120b", alias="MAIN_AGENT_MODEL"
    )
    eval_agent_model: str = Field(
        default="openai/gpt-oss-20b", alias="EVAL_AGENT_MODEL"
    )

    # LLM Endpoints & Credentials
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(
        default="https://api.groq.com/openai/v1", alias="OPENAI_BASE_URL"
    )

    # Chunking Parameters (Source: DesignDocument.md Section 4.1 & Techstack.md Section 2)
    chunk_size_tokens: int = Field(default=600, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=80, alias="CHUNK_OVERLAP_TOKENS")
    max_table_tokens: int = Field(default=1500, alias="MAX_TABLE_TOKENS")
    max_concall_tokens: int = Field(default=700, alias="MAX_CONCALL_TOKENS")
    tiktoken_encoding: str = Field(default="cl100k_base")

    @property
    def raw_pdfs(self) -> Path:
        """Alias for raw_pdfs_dir."""
        return self.raw_pdfs_dir

    @property
    def processed_md(self) -> Path:
        """Alias for processed_md_dir."""
        return self.processed_md_dir

    @property
    def chroma_db(self) -> Path:
        """Alias for chroma_db_dir."""
        return self.chroma_db_dir

    @property
    def bm25_index(self) -> Path:
        """Alias for bm25_index_dir."""
        return self.bm25_index_dir

    def ensure_directories(self) -> None:
        """Create required data subdirectories if they do not exist."""
        for path in [
            self.data_dir,
            self.raw_pdfs_dir,
            self.processed_md_dir,
            self.chroma_db_dir,
            self.bm25_index_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Initialize global settings and automatically create directories on startup
settings = Settings()
settings.ensure_directories()
