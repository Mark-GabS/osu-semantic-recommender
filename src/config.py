"""Configuração central via pydantic-settings (lê variáveis de ambiente e .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    # claude_code: usa o Claude Code local (login da sua conta, sem API key)
    # anthropic_api: SDK oficial com ANTHROPIC_API_KEY | none: 100% determinístico
    llm_backend: Literal["claude_code", "anthropic_api", "none"] = "claude_code"
    llm_enabled: bool = True
    claude_code_bin: str | None = None
    anthropic_api_key: str | None = None
    # vazio = modelo padrão do backend (no Claude Code, o modelo configurado por você;
    # na API, claude-opus-5-5). Aceita aliases do Claude Code: opus, sonnet, haiku.
    planner_model: str = ""
    enrichment_model: str = ""
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_max_tokens: int = 4000
    llm_timeout_s: float = 180.0
    llm_concurrency: int = 4

    # Dados
    songs_dir: Path = PROJECT_ROOT / "data" / "sample_beatmaps"
    data_dir: Path = PROJECT_ROOT / ".osr_data"
    collection_name: str = "beatmaps"

    # Embeddings
    embedding_backend: Literal["hash", "fastembed"] = "hash"
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hash_dim: int = 512

    # Busca
    candidate_pool: int = Field(default=200, description="candidatos recuperados antes do rerank")
    semantic_weight: float = 0.35
    mechanic_weight: float = 0.40
    music_weight: float = 0.25

    # Catálogo online (osu.ppy.sh)
    # auto: API oficial se OSU_CLIENT_ID/SECRET existirem, senão o mirror catboy.best (sem conta)
    catalog_source: Literal["auto", "catboy", "official"] = "auto"
    osu_client_id: str | None = None
    osu_client_secret: str | None = None
    online_status: Literal["ranked", "loved", "qualified", "any"] = "ranked"
    online_max_sets: int = Field(default=300, description="mapsets lidos do catálogo por busca")
    online_fetch_k: int = Field(default=40, description="dificuldades analisadas mecanicamente (.osu baixado)")
    http_concurrency: int = 6
    # pasta Songs local, para não recomendar mapsets que você já tem (vazio = autodetectar)
    owned_songs_dir: Path | None = None

    # Observabilidade
    log_level: str = "INFO"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def enrichment_cache(self) -> Path:
        return self.data_dir / "enrichment_cache.json"

    @property
    def osu_cache_dir(self) -> Path:
        return self.data_dir / "osu_cache"

    @property
    def traces_file(self) -> Path:
        return self.data_dir / "traces.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
