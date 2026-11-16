"""Extração determinística de features mecânicas de um beatmap.

Tudo aqui é cálculo numérico sobre os hit objects: nenhuma chamada a LLM.
As features alimentam (a) os filtros rígidos da busca híbrida, (b) o rerank
mecânico e (c) o prompt de enriquecimento semântico.

Star rating: usa `rosu-pp-py` (port do algoritmo oficial) quando disponível;
caso contrário cai para uma aproximação calibrada, marcada em `sr_source`.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .osu_parser import Beatmap

try:  # dependência opcional
    import rosu_pp_py as _rosu
except Exception:  # pragma: no cover - depende do ambiente
    _rosu = None

SKILLS = (
    "jumps", "tech", "streams", "long_streams", "bursts", "stamina",
    "flow_aim", "high_ar", "sliders", "speed",
)

# divisores de snap comuns (em batidas)
_SNAPS = {1.0: "1/1", 0.5: "1/2", 1 / 3: "1/3", 0.25: "1/4", 1 / 6: "1/6", 0.125: "1/8",
          2.0: "2/1", 0.75: "3/4", 2 / 3: "2/3", 1.5: "3/2"}


class MapFeatures(BaseModel):
    # identidade
    md5: str
    path: str = ""
    title: str
    artist: str
    creator: str
    version: str
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    beatmap_id: int | None = None
    beatmapset_id: int | None = None
    mode: int = 0
    # dificuldade base
    hp: float
    cs: float
    od: float
    ar: float
    # timing
    bpm: float
    bpm_min: float
    bpm_max: float
    length_s: float
    # contagens
    n_objects: int
    n_circles: int
    n_sliders: int
    n_spinners: int
    slider_ratio: float
    notes_per_second: float
    # streams
    max_stream_len: int
    n_bursts: int
    n_streams: int
    n_long_streams: int
    bursts_per_min: float = 0.0
    stream_note_ratio: float
    # aim
    jump_ratio: float
    avg_jump_dist: float
    max_jump_dist: float
    avg_jump_velocity: float
    avg_spacing: float
    # tech
    acute_angle_ratio: float
    angle_std: float
    obtuse_angle_ratio: float
    rhythm_entropy: float
    rhythm_variety: float = 0.0
    sv_change_count: int
    sv_distinct: int = 0
    sv_spread: float = 1.0
    sv_range: float
    # star rating
    star_rating: float
    aim_sr: float | None = None
    speed_sr: float | None = None
    sr_source: str = "approx"
    # scores derivados 0..1
    aim_score: float
    speed_score: float
    stamina_score: float
    tech_score: float
    reading_score: float
    flow_score: float
    skills: list[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.artist} - {self.title} [{self.version}]"

    def skill_score(self, skill: str) -> float:
        return {
            "jumps": self.aim_score,
            "flow_aim": self.flow_score,
            "tech": self.tech_score,
            "streams": self.speed_score,
            "speed": self.speed_score,
            "long_streams": self.stamina_score,
            "stamina": self.stamina_score,
            "bursts": _clamp(self.bursts_per_min / 10),
            "high_ar": self.reading_score,
            "sliders": _clamp((self.slider_ratio - 0.3) / 0.4),
        }.get(skill, 0.0)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _dominant_bpm(bm: Beatmap, end_time: float) -> tuple[float, float, float]:
    reds = [tp for tp in bm.timing_points if tp.uninherited and tp.beat_length > 0]
    if not reds:
        return 120.0, 120.0, 120.0
    durations: Counter[float] = Counter()
    for i, tp in enumerate(reds):
        nxt = reds[i + 1].time if i + 1 < len(reds) else max(end_time, tp.time + 1)
        durations[round(tp.bpm, 1)] += max(0.0, nxt - tp.time)
    bpms = [round(tp.bpm, 1) for tp in reds]
    dominant = durations.most_common(1)[0][0] if durations else bpms[0]
    return dominant, min(bpms), max(bpms)


def _beat_at(bm: Beatmap, t: float) -> float:
    beat = None
    for tp in bm.timing_points:
        if tp.uninherited and tp.beat_length > 0:
            if tp.time > t + 1 and beat is not None:
                break
            beat = tp.beat_length
    return beat or 500.0


def _star_rating(bm: Beatmap) -> tuple[float | None, float | None, float | None]:
    if _rosu is None or bm.path is None:
        return None, None, None
    try:
        rbm = _rosu.Beatmap(path=str(bm.path))
        attrs = _rosu.Difficulty().calculate(rbm)
        return float(attrs.stars), float(getattr(attrs, "aim", 0) or 0), float(getattr(attrs, "speed", 0) or 0)
    except Exception:
        return None, None, None


def extract_features(bm: Beatmap) -> MapFeatures:
    objs = [o for o in bm.hit_objects if not o.is_spinner]
    n_sp = sum(1 for o in bm.hit_objects if o.is_spinner)
    n_sl = sum(1 for o in objs if o.is_slider)
    n_ci = len(objs) - n_sl
    first = bm.hit_objects[0].time if bm.hit_objects else 0
    last = max((o.end_time for o in bm.hit_objects), default=0)
    length_s = max(0.001, (last - first) / 1000)
    bpm, bpm_min, bpm_max = _dominant_bpm(bm, last)

    radius = 54.4 - 4.48 * bm.cs
    jump_threshold = 1.6 * (2 * radius)  # >= 1.6 diâmetros de círculo

    # ---- transições ---------------------------------------------------------
    run = 1
    runs: list[int] = []
    snap_counter: Counter[str] = Counter()
    rhythm_counter: Counter[str] = Counter()
    spacings: list[float] = []
    jump_dists: list[float] = []
    jump_vels: list[float] = []
    aim_vels: list[float] = []  # velocidade de cursor em movimentos 1/2..1/1 (fora de streams)
    n_trans = 0
    for a, b in zip(objs, objs[1:]):
        delta = b.time - a.time
        if delta <= 0:
            continue
        n_trans += 1
        beat = _beat_at(bm, a.time)
        rel = delta / beat
        snap = min(_SNAPS, key=lambda s: abs(s - rel))
        if abs(snap - rel) < 0.06:
            snap_counter[_SNAPS[snap]] += 1
            # duração de slider distorce o intervalo início->início: ritmo só a partir de círculos
            if a.is_circle and rel <= 2.05:
                rhythm_counter[_SNAPS[snap]] += 1
        else:
            snap_counter["other"] += 1

        # stream: 1/4 ou mais rápido, com teto absoluto (evita 1/4 em BPM muito baixo)
        if a.is_circle and rel <= 0.25 * 1.15 and delta <= 200:
            run += 1
        else:
            if run >= 2:
                runs.append(run)
            run = 1

        dist = math.hypot(b.x - a.x, b.y - a.y)
        spacings.append(dist)
        if 0.4 <= rel <= 1.1:
            aim_vels.append(dist / delta)
        # jump: 1/2 a 1/1, distância >= 1.6 diâmetros e velocidade de cursor relevante
        if a.is_circle and 0.5 * 0.85 <= rel <= 1.1 and dist >= jump_threshold and dist / delta >= 0.62:
            jump_dists.append(dist)
            jump_vels.append(dist / delta)
    if run >= 2:
        runs.append(run)

    n_bursts = sum(1 for r in runs if 3 <= r <= 6)
    n_streams = sum(1 for r in runs if 7 <= r <= 15)
    n_long = sum(1 for r in runs if r >= 16)
    stream_notes = sum(r for r in runs if r >= 7)
    max_stream = max(runs, default=1)
    if max_stream < 3:
        max_stream = 0 if not runs else max_stream

    # ---- ângulos (apenas movimentos "de aim", fora de streams) ---------------
    acute = obtuse = n_ang = 0
    angles: list[float] = []
    for a, b, c in zip(objs, objs[1:], objs[2:]):
        v1 = (b.x - a.x, b.y - a.y)
        v2 = (c.x - b.x, c.y - b.y)
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 60 or n2 < 60:
            continue
        cos = (-v1[0] * v2[0] - v1[1] * v2[1]) / (n1 * n2)  # ângulo no vértice b
        ang = math.degrees(math.acos(_clamp(cos, -1, 1)))
        n_ang += 1
        angles.append(ang)
        if ang < 60:
            acute += 1
        elif ang > 110:
            obtuse += 1
    acute_ratio = acute / n_ang if n_ang else 0.0
    obtuse_ratio = obtuse / n_ang if n_ang else 0.0
    if len(angles) > 1:
        mean_ang = sum(angles) / len(angles)
        angle_std = math.sqrt(sum((a - mean_ang) ** 2 for a in angles) / len(angles))
    else:
        angle_std = 0.0

    # ---- ritmo e SV -------------------------------------------------------------
    total_snaps = sum(snap_counter.values())
    entropy = 0.0
    if total_snaps:
        for c in snap_counter.values():
            p = c / total_snaps
            entropy -= p * math.log(p)
        entropy /= math.log(len(_SNAPS) + 1)
    # mistura de ritmos: fração de notas FORA dos dois snaps mais comuns. Um mapa todo em
    # 1/3 é consistente (variedade ~0); tech de ritmo mistura 1/2, 1/3, 1/4, 3/4...
    total_rhythm = sum(rhythm_counter.values())
    top2 = sum(c for _, c in rhythm_counter.most_common(2))
    rhythm_variety = 1 - top2 / total_rhythm if total_rhythm >= 20 else 0.0
    svs = [tp.sv_multiplier for tp in bm.timing_points if not tp.uninherited]
    sv_changes = sum(1 for a, b in zip([1.0, *svs], svs) if abs(a - b) > 0.05)
    sv_range = (max(svs) - min(min(svs), 1.0)) if svs else 0.0
    sv_set = {round(v * 20) / 20 for v in svs} | {1.0}
    sv_distinct = len(sv_set) - 1
    sv_spread = max(sv_set) / min(sv_set)

    n = max(1, n_trans)
    jump_ratio = len(jump_dists) / n
    avg_jump_vel = sum(jump_vels) / len(jump_vels) if jump_vels else 0.0
    stream_ratio = stream_notes / max(1, len(objs))

    # ---- scores 0..1 ------------------------------------------------------------
    aim_score = 0.5 * _clamp(jump_ratio / 0.6) + 0.5 * _clamp(avg_jump_vel / 1.8)
    speed_score = 0.6 * _clamp(stream_ratio / 0.5) + 0.4 * _clamp((bpm - 140) / 100) * (stream_ratio > 0.05)
    stamina_score = 0.6 * _clamp(max_stream / 48) + 0.4 * _clamp(n_long / 6)
    # tech = ritmo misturado + SV realmente variado + ângulos irregulares.
    # Constantes calibradas em mapas reais (ranked) + sintéticos: em mapas reais quase todo
    # mapa tem alguma mudança de SV e angle_std entre 35 e 48, então só os extremos contam.
    rhythm_part = _clamp(rhythm_variety / 0.2)
    sv_part = _clamp((sv_spread - 1.4) / 2.0) * _clamp(sv_distinct / 5)
    angle_part = _clamp((angle_std - 40) / 10)
    tech_score = 0.45 * rhythm_part + 0.35 * sv_part + 0.20 * angle_part
    reading_score = max(_clamp((bm.ar - 9.3) / 0.9), _clamp((7.5 - bm.ar) / 3) * _clamp(len(objs) / length_s / 4))
    avg_sp = sum(spacings) / len(spacings) if spacings else 0.0
    # flow: movimento contínuo (ângulos abertos) com spacing acima do normal; em mapas reais
    # obtuse_ratio ~0.32 e spacing ~127px são a mediana, então medimos o excesso sobre isso
    # ... e com velocidade de cursor de verdade: Easy/Normal têm ângulos abertos e spacing
    # grande, mas o cursor anda devagar, então não treinam flow aim
    avg_aim_vel = sum(aim_vels) / len(aim_vels) if aim_vels else 0.0
    flow_score = (_clamp((obtuse_ratio - 0.2) / 0.25) * (1 - _clamp(stream_ratio / 0.4))
                  * _clamp((avg_sp - 70) / 60) * _clamp((avg_aim_vel - 0.25) / 0.35))
    bursts_per_min = n_bursts / (length_s / 60)

    # ---- star rating --------------------------------------------------------------
    sr, aim_sr, speed_sr = _star_rating(bm)
    sr_source = "rosu-pp"
    if sr is None:
        sr_source = "approx"
        density = len(objs) / length_s
        aim_part = 1.9 * math.sqrt(max(0.0, avg_jump_vel)) * (0.4 + _clamp(jump_ratio / 0.6))
        speed_part = (bpm / 75) * (0.35 + _clamp(stream_ratio / 0.4)) * _clamp(density / 4)
        sr = 0.6 + (aim_part ** 1.1 + speed_part ** 1.1) ** (1 / 1.1) * (1 + (bm.cs - 4) * 0.05)

    skills_map = {
        "jumps": aim_score >= 0.5,
        "tech": tech_score >= 0.45,
        "streams": n_streams + n_long >= 2 or stream_ratio >= 0.25,
        "long_streams": n_long >= 1,
        "bursts": bursts_per_min >= 6,  # mediana real ~2.5/min
        "stamina": stamina_score >= 0.5,
        "flow_aim": flow_score >= 0.35,
        "high_ar": bm.ar >= 9.7,
        "sliders": n_sl / max(1, len(objs)) >= 0.6,  # mediana real ~0.49
        "speed": speed_score >= 0.55,
    }

    return MapFeatures(
        md5=bm.md5,
        path=str(bm.path) if bm.path else "",
        title=bm.title,
        artist=bm.artist,
        creator=bm.creator,
        version=bm.version,
        source=bm.source,
        tags=bm.tags,
        beatmap_id=bm.beatmap_id,
        beatmapset_id=bm.beatmapset_id,
        mode=bm.mode,
        hp=bm.hp, cs=bm.cs, od=bm.od, ar=bm.ar,
        bpm=round(bpm, 2), bpm_min=round(bpm_min, 2), bpm_max=round(bpm_max, 2),
        length_s=round(length_s, 2),
        n_objects=len(bm.hit_objects), n_circles=n_ci, n_sliders=n_sl, n_spinners=n_sp,
        slider_ratio=round(n_sl / max(1, len(objs)), 4),
        notes_per_second=round(len(objs) / length_s, 3),
        max_stream_len=max_stream, n_bursts=n_bursts, n_streams=n_streams, n_long_streams=n_long,
        bursts_per_min=round(bursts_per_min, 2),
        stream_note_ratio=round(stream_ratio, 4),
        jump_ratio=round(jump_ratio, 4),
        avg_jump_dist=round(sum(jump_dists) / len(jump_dists), 2) if jump_dists else 0.0,
        max_jump_dist=round(max(jump_dists, default=0.0), 2),
        avg_jump_velocity=round(avg_jump_vel, 4),
        avg_spacing=round(sum(spacings) / len(spacings), 2) if spacings else 0.0,
        acute_angle_ratio=round(acute_ratio, 4),
        angle_std=round(angle_std, 2),
        obtuse_angle_ratio=round(obtuse_ratio, 4),
        rhythm_entropy=round(entropy, 4),
        rhythm_variety=round(rhythm_variety, 4),
        sv_change_count=sv_changes,
        sv_distinct=sv_distinct,
        sv_spread=round(sv_spread, 3),
        sv_range=round(sv_range, 3),
        star_rating=round(sr, 2),
        aim_sr=round(aim_sr, 2) if aim_sr is not None else None,
        speed_sr=round(speed_sr, 2) if speed_sr is not None else None,
        sr_source=sr_source,
        aim_score=round(aim_score, 3),
        speed_score=round(speed_score, 3),
        stamina_score=round(stamina_score, 3),
        tech_score=round(tech_score, 3),
        reading_score=round(reading_score, 3),
        flow_score=round(flow_score, 3),
        skills=[k for k, v in skills_map.items() if v],
    )


def features_from_path(path: str | Path) -> MapFeatures:
    from .osu_parser import parse_osu_file

    return extract_features(parse_osu_file(path))
