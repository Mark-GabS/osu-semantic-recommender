import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402

SAMPLES = ROOT / "data" / "sample_beatmaps"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        anthropic_api_key=None,
        llm_enabled=False,
        llm_backend="none",
        data_dir=tmp_path / "osr",
        songs_dir=SAMPLES,
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        embedding_backend="hash",
    )


@pytest.fixture
def sample_files() -> list[Path]:
    files = sorted(SAMPLES.glob("*.osu"))
    assert files, "rode `python data/generate_samples.py` para gerar os mapas de exemplo"
    return files
