"""Enriquecimento semântico de beatmaps.

Divisão de responsabilidades (decisão central da arquitetura):
- *Mecânica* vem de números determinísticos (`MapFeatures`). O LLM recebe essas
  métricas e é instruído a NÃO contradizê-las.
- *Semântica* (gênero, mood, estilo de jogo descrito em linguagem natural, para
  que tipo de treino o mapa serve) é onde o LLM agrega valor: ele conhece
  artistas/animes/fontes e traduz números em descrições pesquisáveis.

Saída via Structured Outputs (`MapEnrichment`). Sem LLM, um enriquecedor por
template gera o mesmo schema a partir de tags/metadados + features, o que deixa
o pipeline 100% funcional offline. Resultados ficam em cache por MD5.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..llm import LLMAuthError, LLMClient, LLMError
from ..observability import get_logger, span
from ..parser.feature_extractor import MapFeatures

log = get_logger("enrichment")

ENRICHMENT_VERSION = 3  # v3: métricas tech/flow/sliders recalibradas em mapas reais


class MapEnrichment(BaseModel):
    summary: str = Field(description="2-3 frases em inglês descrevendo música e jogabilidade")
    genres: list[str] = Field(description="gêneros musicais canônicos em inglês minúsculo, ex: 'jrock', 'electronic'")
    moods: list[str] = Field(description="moods em inglês minúsculo, ex: 'energetic', 'calm', 'aggressive'")
    play_style: list[str] = Field(description="descritores de jogabilidade, ex: 'technical jumps', 'flow aim'")
    training_focus: list[str] = Field(description="habilidades que o mapa treina, ex: 'aim', 'speed', 'stamina'")
    energy: Literal["low", "medium", "high"]
    source: str = Field(default="template", description="preenchido pelo sistema: 'llm:<modelo>' ou 'template'")


SYSTEM_PROMPT = """You are an osu!standard beatmap analyst that writes search-index metadata.

You receive one beatmap: song metadata (artist, title, source, mapper tags) and
DETERMINISTIC mechanical metrics computed from the hit objects.

Rules:
- Mechanics: describe play style ONLY from the metrics. Never claim streams if
  max_stream_len is small, never claim jumps if aim_score is low, etc.
  Metric guide: aim_score/speed_score/stamina_score/tech_score/flow_score/reading_score are 0..1.
  max_stream_len = longest run of 1/4 notes; n_long_streams = runs >= 16 notes.
- Music: infer genres and moods from artist, title, source and tags, using your
  knowledge of artists/anime/games when you recognize them. If unknown, rely on tags
  and BPM; prefer fewer, confident labels.
- Use lowercase English canonical labels. Genres examples: jrock, rock, metal, jpop,
  anime, vocaloid, electronic, classical, piano, lofi, hip hop.
- summary: 2-3 sentences in English, concrete and searchable.
- Leave `source` as "template"; the system overwrites it."""


# ---------------------------------------------------------------------------
# Enriquecimento por template (offline)
# ---------------------------------------------------------------------------
GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "jrock": ("j-rock", "jrock", "japanese rock"),
    "rock": ("rock", "guitar", "band", "j-rock", "jrock"),
    "metal": ("metal", "j-metal", "heavy"),
    "jpop": ("j-pop", "jpop", "idol", "japanese pop"),
    "anime": ("anime", "opening", "ending", "op", "ed"),
    "vocaloid": ("vocaloid", "miku", "hatsune"),
    "electronic": ("electronic", "edm", "trance", "hardstyle", "glitch", "dubstep", "techno", "house"),
    "classical": ("classical", "orchestral", "waltz", "symphony"),
    "piano": ("piano",),
    "lofi": ("lofi", "lo-fi", "chill"),
    "hip hop": ("hip", "hiphop", "rap"),
}
MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "energetic": ("energetic", "upbeat", "fast", "hype"),
    "calm": ("calm", "relaxing", "chill", "soft"),
    "aggressive": ("aggressive", "heavy", "intense", "brutal"),
    "happy": ("happy", "cute", "idol", "upbeat"),
    "emotional": ("emotional", "sad", "melancholic"),
    "elegant": ("elegant", "waltz", "orchestral"),
}


def _match(text: str, table: dict[str, tuple[str, ...]]) -> list[str]:
    words = set(text.split())
    out = []
    for label, kws in table.items():
        if any((kw in words) if " " not in kw else (kw in text) for kw in kws):
            out.append(label)
    return out


def template_enrichment(f: MapFeatures) -> MapEnrichment:
    text = " ".join([f.artist, f.title, f.source, *f.tags]).lower()
    genres = _match(text, GENRE_KEYWORDS)
    moods = _match(text, MOOD_KEYWORDS)
    if f.bpm >= 170 and "energetic" not in moods and "calm" not in moods:
        moods.append("energetic")
    if f.bpm <= 125 and not moods:
        moods.append("calm")

    style: list[str] = []
    if "jumps" in f.skills:
        style.append("technical jumps" if f.tech_score >= 0.4 else "jumps")
        if f.avg_jump_dist >= 220:
            style.append("wide spacing")
    if "flow_aim" in f.skills:
        style.append("flow aim")
    if "long_streams" in f.skills:
        style.append("long streams" if f.max_stream_len < 32 else "deathstreams")
    elif "streams" in f.skills:
        style.append("short streams")
    if "bursts" in f.skills:
        style.append("bursts")
    if f.tech_score >= 0.4 and "jumps" not in f.skills:
        style.append("technical rhythm")
    if f.sv_change_count >= 3:
        style.append("slider velocity changes")
    if "sliders" in f.skills:
        style.append("slider heavy")
    if f.ar >= 9.7:
        style.append("high ar reading")
    if f.max_stream_len < 7:
        style.append("no streams")
    focus_scores = {
        "aim": f.aim_score, "speed": f.speed_score, "stamina": f.stamina_score,
        "tech": f.tech_score, "reading": f.reading_score, "flow": f.flow_score,
    }
    focus = [k for k, v in sorted(focus_scores.items(), key=lambda kv: -kv[1]) if v >= 0.4][:3] or ["consistency"]
    energy = "high" if f.bpm >= 175 or "energetic" in moods or "aggressive" in moods else (
        "low" if f.bpm < 130 or "calm" in moods else "medium")
    g = ", ".join(genres) or "unknown genre"
    summary = (
        f"{f.artist} - {f.title}: {g} track at {f.bpm:.0f} BPM ({', '.join(moods) or 'neutral'} mood). "
        f"{f.star_rating:.2f} star map focused on {', '.join(style) or 'basic patterns'}; "
        f"good for training {', '.join(focus)}."
    )
    return MapEnrichment(summary=summary, genres=genres, moods=moods, play_style=style,
                         training_focus=focus, energy=energy, source="template")


# ---------------------------------------------------------------------------
# Enriquecimento via LLM, com cache
# ---------------------------------------------------------------------------
def _features_for_prompt(f: MapFeatures) -> dict:
    keep = [
        "artist", "title", "version", "source", "tags", "star_rating", "bpm", "length_s", "ar", "cs", "od",
        "notes_per_second", "max_stream_len", "n_bursts", "n_streams", "n_long_streams", "stream_note_ratio",
        "jump_ratio", "avg_jump_dist", "sv_change_count", "slider_ratio", "aim_score", "speed_score",
        "stamina_score", "tech_score", "flow_score", "reading_score", "skills",
    ]
    data = f.model_dump()
    return {k: data[k] for k in keep}


class Enricher:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None, use_llm: bool = True):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.use_llm = use_llm and self.llm.available
        self.cache_path: Path = self.settings.enrichment_cache
        self._cache: dict[str, dict] = self._load_cache()
        self.stats = {"cache_hits": 0, "llm": 0, "template": 0, "llm_errors": 0}
        self._lock = threading.Lock()  # enrich() pode rodar em paralelo (LLM_CONCURRENCY)

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _load_cache(self) -> dict[str, dict]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def _key(self, f: MapFeatures) -> str:
        mode = self.llm.label(self.settings.enrichment_model) if self.use_llm else "template"
        return f"{f.md5}:{mode}:v{ENRICHMENT_VERSION}"

    def enrich(self, f: MapFeatures) -> MapEnrichment:
        key = self._key(f)
        if key in self._cache:
            self._count("cache_hits")
            return MapEnrichment.model_validate(self._cache[key])
        with span("enrichment.map", md5=f.md5, llm=self.use_llm) as sp:
            result: MapEnrichment | None = None
            if self.use_llm:
                try:
                    result = self.llm.structured(
                        model=self.settings.enrichment_model,
                        system=SYSTEM_PROMPT,
                        user="Beatmap:\n" + json.dumps(_features_for_prompt(f), ensure_ascii=False, indent=1),
                        schema=MapEnrichment,
                        purpose="enrichment",
                    )
                    result.source = self.llm.label(self.settings.enrichment_model)
                    self._count("llm")
                except LLMAuthError:
                    self._count("llm_errors")  # já logado uma vez pelo circuit breaker
                except LLMError as exc:
                    self._count("llm_errors")
                    log.warning("LLM falhou para %s (%s); usando template", f.display_name, exc)
            if result is None:
                result = template_enrichment(f)
                self._count("template")
            sp.set(source=result.source)
        # só cacheia o que veio do modo pedido (falha de LLM não envenena o cache)
        if not self.use_llm or result.source.startswith("llm:"):
            self._cache[key] = result.model_dump()
        return result


def build_document(f: MapFeatures, e: MapEnrichment) -> str:
    """Texto indexado no banco vetorial (o que a busca semântica 'enxerga')."""
    return (
        f"{f.artist} - {f.title} [{f.version}]. Source: {f.source or '-'}. "
        f"Genres: {', '.join(e.genres) or 'unknown'}. Moods: {', '.join(e.moods) or 'neutral'}. "
        f"Energy: {e.energy}. Play style: {', '.join(e.play_style)}. "
        f"Training: {', '.join(e.training_focus)}. {e.summary} Tags: {' '.join(f.tags)}"
    )
