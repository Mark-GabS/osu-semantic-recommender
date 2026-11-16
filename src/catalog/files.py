"""Arquivos de beatmaps online: .osu (para análise mecânica), .osz (download) e mapas já instalados."""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..observability import get_logger, span
from .models import OnlineSet
from .sources import USER_AGENT

log = get_logger("files")

OSU_FILE_URL = "https://osu.ppy.sh/osu/{beatmap_id}"
OSZ_URL = "https://osu.direct/api/d/{set_id}"


class OsuFileCache:
    """Baixa o .osu de cada dificuldade (endpoint público do osu!) com cache em disco."""

    def __init__(self, cache_dir: Path, client: httpx.Client | None = None, concurrency: int = 6):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=40, follow_redirects=True)
        self.concurrency = concurrency
        self.stats = {"cache_hits": 0, "downloaded": 0, "failed": 0}

    def path_for(self, beatmap_id: int) -> Path:
        return self.dir / f"{beatmap_id}.osu"

    def get(self, beatmap_id: int, checksum: str | None = None) -> Path | None:
        path = self.path_for(beatmap_id)
        if path.exists():
            if checksum is None or hashlib.md5(path.read_bytes()).hexdigest() == checksum:
                self.stats["cache_hits"] += 1
                return path
        try:
            r = self.client.get(OSU_FILE_URL.format(beatmap_id=beatmap_id))
            r.raise_for_status()
            if not r.content.lstrip().startswith(b"osu file format"):
                raise ValueError("resposta não é um .osu")
        except (httpx.HTTPError, ValueError) as exc:
            self.stats["failed"] += 1
            log.debug("falha baixando .osu %s: %s", beatmap_id, exc)
            return None
        path.write_bytes(r.content)
        self.stats["downloaded"] += 1
        return path

    def get_many(self, items: list[tuple[int, str | None]]) -> dict[int, Path | None]:
        with span("catalog.osu_files", n=len(items)) as sp:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                paths = list(pool.map(lambda it: self.get(*it), items))
            sp.set(**self.stats)
        return {bid: p for (bid, _), p in zip(items, paths)}


_SAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def download_osz(s: OnlineSet, dest_dir: Path, client: httpx.Client | None = None) -> Path:
    """Baixa o .osz de um mapset (mirror osu.direct). Pula se já existir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = _SAFE.sub("_", f"{s.id} {s.artist} - {s.title}")[:150] + ".osz"
    out = dest_dir / name
    if out.exists() and out.stat().st_size > 0:
        return out
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True)
    tmp = out.with_suffix(".part")
    with client.stream("GET", OSZ_URL.format(set_id=s.id)) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 16):
                fh.write(chunk)
    if tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download do mapset {s.id} veio vazio")
    tmp.replace(out)
    return out


def detect_songs_dir() -> Path | None:
    """Pasta Songs do osu!stable (Windows ou Linux/Wine), se existir."""
    candidates = []
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "osu!" / "Songs")
    home = Path.home()
    candidates += [home / ".local/share/osu-wine/osu!/Songs", home / ".wine/drive_c/osu!/Songs"]
    return next((c for c in candidates if c.is_dir()), None)


def owned_set_ids(songs_dir: Path | None) -> set[int]:
    """IDs de mapsets já instalados (pastas da Songs começam com o id: '123456 Artista - Título')."""
    if not songs_dir or not Path(songs_dir).is_dir():
        return set()
    out = set()
    for entry in Path(songs_dir).iterdir():
        m = re.match(r"^(\d+)\s", entry.name)
        if m and entry.is_dir():
            out.add(int(m[1]))
    return out
