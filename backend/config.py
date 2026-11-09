"""
config.py — Central configuration using pydantic-settings.
All settings are loaded from .env file or environment variables.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "legalrag123"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Paths
    data_dir: str = "./data"
    raw_dir: str = "./data/raw"
    structured_dir: str = "./data/structured"
    pdf_path: str = "./document.pdf"

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_vector: int = 10
    top_k_graph: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        """Create all necessary data directories."""
        for d in [self.data_dir, self.raw_dir, self.structured_dir, self.chroma_persist_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
