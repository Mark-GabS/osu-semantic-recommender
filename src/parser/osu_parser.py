"""Parser determinístico do formato .osu (osu!standard, formato v3..v14).

Sem dependências externas e sem LLM: o objetivo é ser rápido, previsível e
testável. Extrai metadados, dificuldade, timing points e hit objects, além do
MD5 do arquivo (identificador usado pelo `collection.db` do osu!).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# bits do campo "type" de um hit object
CIRCLE, SLIDER, NEW_COMBO, SPINNER, HOLD = 1, 2, 4, 8, 128


class OsuParseError(ValueError):
    pass


@dataclass(slots=True)
class TimingPoint:
    time: float
    beat_length: float
    meter: int
    uninherited: bool

    @property
    def bpm(self) -> float | None:
        return 60000.0 / self.beat_length if self.uninherited and self.beat_length > 0 else None

    @property
    def sv_multiplier(self) -> float:
        """Multiplicador de slider velocity (pontos herdados usam beat_length negativo)."""
        if self.uninherited or self.beat_length >= 0:
            return 1.0
        return max(0.1, min(10.0, -100.0 / self.beat_length))


@dataclass(slots=True)
class HitObject:
    x: float
    y: float
    time: int
    type: int
    end_time: int
    slider_length: float = 0.0
    slider_repeats: int = 1

    @property
    def is_circle(self) -> bool:
        return bool(self.type & CIRCLE)

    @property
    def is_slider(self) -> bool:
        return bool(self.type & SLIDER)

    @property
    def is_spinner(self) -> bool:
        return bool(self.type & SPINNER)

    @property
    def new_combo(self) -> bool:
        return bool(self.type & NEW_COMBO)


@dataclass(slots=True)
class Beatmap:
    path: Path | None
    md5: str
    format_version: int
    mode: int
    title: str
    title_unicode: str
    artist: str
    artist_unicode: str
    creator: str
    version: str
    source: str
    tags: list[str]
    beatmap_id: int | None
    beatmapset_id: int | None
    hp: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float
    slider_tick_rate: float
    timing_points: list[TimingPoint] = field(default_factory=list)
    hit_objects: list[HitObject] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.artist} - {self.title} [{self.version}]"


def _kv(line: str) -> tuple[str, str]:
    key, _, value = line.partition(":")
    return key.strip(), value.strip()


def _float(d: dict[str, str], key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except ValueError:
        return default


def _int_or_none(value: str | None) -> int | None:
    try:
        v = int(value) if value not in (None, "") else None
    except ValueError:
        return None
    return v if v and v > 0 else None


def parse_osu_bytes(raw: bytes, path: Path | None = None) -> Beatmap:
    md5 = hashlib.md5(raw).hexdigest()
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines or not lines[0].strip().startswith("osu file format v"):
        raise OsuParseError(f"cabeçalho .osu inválido: {path}")
    try:
        fmt = int(lines[0].strip().removeprefix("osu file format v"))
    except ValueError:
        fmt = 14

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1]
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(s)

    general = dict(_kv(l) for l in sections.get("General", []))
    meta = dict(_kv(l) for l in sections.get("Metadata", []))
    diff = dict(_kv(l) for l in sections.get("Difficulty", []))

    od = _float(diff, "OverallDifficulty", 5.0)
    timing = [_parse_timing(l) for l in sections.get("TimingPoints", [])]
    timing = sorted((t for t in timing if t is not None), key=lambda t: t.time)

    bm = Beatmap(
        path=path,
        md5=md5,
        format_version=fmt,
        mode=int(_float(general, "Mode", 0)),
        title=meta.get("Title", ""),
        title_unicode=meta.get("TitleUnicode", meta.get("Title", "")),
        artist=meta.get("Artist", ""),
        artist_unicode=meta.get("ArtistUnicode", meta.get("Artist", "")),
        creator=meta.get("Creator", ""),
        version=meta.get("Version", ""),
        source=meta.get("Source", ""),
        tags=[t for t in meta.get("Tags", "").split() if t],
        beatmap_id=_int_or_none(meta.get("BeatmapID")),
        beatmapset_id=_int_or_none(meta.get("BeatmapSetID")),
        hp=_float(diff, "HPDrainRate", 5.0),
        cs=_float(diff, "CircleSize", 5.0),
        od=od,
        # mapas antigos não têm AR: o padrão do jogo é AR = OD
        ar=_float(diff, "ApproachRate", od),
        slider_multiplier=_float(diff, "SliderMultiplier", 1.4),
        slider_tick_rate=_float(diff, "SliderTickRate", 1.0),
        timing_points=timing,
    )
    bm.hit_objects = _parse_hit_objects(sections.get("HitObjects", []), bm)
    return bm


def parse_osu_file(path: str | Path) -> Beatmap:
    p = Path(path)
    return parse_osu_bytes(p.read_bytes(), p)


def _parse_timing(line: str) -> TimingPoint | None:
    parts = line.split(",")
    if len(parts) < 2:
        return None
    try:
        time = float(parts[0])
        beat_length = float(parts[1])
        meter = int(parts[2]) if len(parts) > 2 and parts[2] else 4
        uninherited = (parts[6].strip() == "1") if len(parts) > 6 else beat_length > 0
    except ValueError:
        return None
    return TimingPoint(time, beat_length, meter, uninherited)


def _timing_at(points: list[TimingPoint], t: float) -> tuple[float, float]:
    """(beat_length do ponto vermelho ativo, multiplicador SV ativo) no instante t."""
    beat, sv = 500.0, 1.0
    for tp in points:
        if tp.time > t + 1:
            break
        if tp.uninherited:
            beat, sv = tp.beat_length, 1.0
        else:
            sv = tp.sv_multiplier
    if beat <= 0:
        beat = 500.0
    return beat, sv


def _parse_hit_objects(lines: list[str], bm: Beatmap) -> list[HitObject]:
    objs: list[HitObject] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            x, y, t, typ = float(parts[0]), float(parts[1]), int(float(parts[2])), int(parts[3])
        except ValueError:
            continue
        end = t
        length, repeats = 0.0, 1
        if typ & SLIDER and len(parts) >= 8:
            try:
                repeats = max(1, int(parts[6]))
                length = float(parts[7])
            except ValueError:
                pass
            beat, sv = _timing_at(bm.timing_points, t)
            px_per_beat = bm.slider_multiplier * 100.0 * sv
            if px_per_beat > 0:
                end = int(t + length / px_per_beat * beat * repeats)
        elif typ & (SPINNER | HOLD) and len(parts) >= 6:
            try:
                end = int(float(parts[5].split(":")[0]))
            except ValueError:
                pass
        objs.append(HitObject(x, y, t, typ, end, length, repeats))
    objs.sort(key=lambda o: o.time)
    return objs


def iter_osu_files(root: str | Path):
    """Itera recursivamente por arquivos .osu (pasta Songs inteira ou subpasta)."""
    yield from sorted(Path(root).rglob("*.osu"))
