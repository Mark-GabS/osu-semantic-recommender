"""Gera beatmaps .osu sintéticos (determinísticos) em data/sample_beatmaps/.

Os mapas cobrem perfis mecânicos distintos (jumps técnicos, streams longas,
flow aim, bursts, tech de sliders, leitura em AR alto...) e metadados variados
(gênero/mood nas tags), para que parser, indexação, busca e evals rodem offline.

Artistas e músicas são fictícios.

Uso:  python data/generate_samples.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sample_beatmaps"
W, H = 512, 384


@dataclass
class Spec:
    slug: str
    artist: str
    title: str
    version: str
    tags: str
    source: str
    bpm: float
    ar: float
    cs: float
    od: float
    hp: float
    recipe: list[tuple[str, dict]]
    seconds: int = 100
    seed: int = 0
    sv_changes: list[tuple[float, float]] = field(default_factory=list)  # (frac do mapa, multiplicador SV)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Builder:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.beat = 60000.0 / spec.bpm
        self.t = 1000.0
        self.x, self.y = 256.0, 192.0
        self.angle = 0.0
        self.objects: list[str] = []
        self.new_combo = True

    # ---- primitivas ---------------------------------------------------
    def _type(self, base: int) -> int:
        t = base | (4 if self.new_combo else 0)
        self.new_combo = False
        return t

    def circle(self, x: float, y: float) -> None:
        self.x, self.y = clamp(x, 0, W), clamp(y, 0, H)
        self.objects.append(f"{int(self.x)},{int(self.y)},{int(self.t)},{self._type(1)},0,0:0:0:0:")

    def slider(self, x: float, y: float, length: float, angle: float, beats: float) -> None:
        self.x, self.y = clamp(x, 0, W), clamp(y, 0, H)
        ex = clamp(self.x + math.cos(angle) * length, 0, W)
        ey = clamp(self.y + math.sin(angle) * length, 0, H)
        self.objects.append(
            f"{int(self.x)},{int(self.y)},{int(self.t)},{self._type(2)},0,"
            f"L|{int(ex)}:{int(ey)},1,{length:.1f}"
        )
        self.x, self.y = ex, ey
        return beats

    def step(self, dist: float, turn: float) -> tuple[float, float]:
        self.angle += turn
        nx, ny = self.x + math.cos(self.angle) * dist, self.y + math.sin(self.angle) * dist
        # rebate nas bordas
        if not (32 <= nx <= W - 32):
            self.angle = math.pi - self.angle
            nx = self.x + math.cos(self.angle) * dist
        if not (32 <= ny <= H - 32):
            self.angle = -self.angle
            ny = self.y + math.sin(self.angle) * dist
        return clamp(nx, 16, W - 16), clamp(ny, 16, H - 16)

    # ---- seções -------------------------------------------------------
    def jumps(self, beats: int = 16, dist: float = 260, tech: float = 0.2, div: int = 2) -> None:
        """Jumps em 1/div. `tech` aumenta ângulos agudos, variação de spacing e ritmos 1/3."""
        self.new_combo = True
        n = beats * div
        for i in range(n):
            turn = math.pi * (0.6 + self.rng.uniform(-0.05, 0.05))
            if self.rng.random() < tech:
                turn = math.pi * self.rng.choice([0.2, 0.85, 1.0, -0.3])
            d = dist * (1 + self.rng.uniform(-tech, tech) * 0.6)
            self.circle(*self.step(d, turn))
            gap = self.beat / div
            if self.rng.random() < tech * 0.5:
                gap = self.beat * self.rng.choice([1 / 3, 2 / 3, 3 / 4, 1 / 4])
            self.t += gap
            if i % 8 == 7:
                self.new_combo = True

    def flow(self, beats: int = 16, dist: float = 180) -> None:
        self.new_combo = True
        for i in range(beats * 2):
            self.circle(*self.step(dist, math.pi * 0.18))
            self.t += self.beat / 2
            if i % 8 == 7:
                self.new_combo = True

    def stream(self, notes: int = 16, dist: float = 22, gap_after: float = 1.0) -> None:
        self.new_combo = True
        for _ in range(notes):
            self.circle(*self.step(dist, 0.22))
            self.t += self.beat / 4
        self.t += self.beat * gap_after - self.beat / 4

    def bursts(self, count: int = 6, size: int = 4, dist: float = 30, jump: float = 200) -> None:
        for _ in range(count):
            self.stream(size, dist, gap_after=0.5)
            self.circle(*self.step(jump, math.pi * 0.6))
            self.t += self.beat / 2

    def sliders(self, count: int = 12, length: float = 120, tech: float = 0.0) -> None:
        self.new_combo = True
        for i in range(count):
            ang = self.rng.uniform(0, 2 * math.pi)
            x, y = self.step(90 + 80 * tech, math.pi * 0.5)
            ln = length * (1 + self.rng.uniform(-tech, tech) * 0.5)
            self.slider(x, y, ln, ang, 1)
            # duração conservadora (assume SV mínimo 0.6) para nunca sobrepor objetos
            beats = math.ceil(ln / (180 * 0.6) * 4) / 4
            dur = beats * self.beat
            self.t += dur + self.beat / 2 * (1 if self.rng.random() > tech else 0.5)
            if i % 4 == 3:
                self.new_combo = True

    def rest(self, beats: float = 2) -> None:
        self.t += self.beat * beats
        self.new_combo = True

    def build(self) -> None:
        end = 1000 + self.spec.seconds * 1000
        while self.t < end:
            for name, kw in self.spec.recipe:
                getattr(self, name)(**kw)
                if self.t >= end:
                    break


def render(spec: Spec) -> str:
    b = Builder(spec)
    b.build()
    end = b.t
    timing = [f"1000,{b.beat:.6f},4,2,1,60,1,0"]
    for frac, mult in spec.sv_changes:
        t = int(1000 + (end - 1000) * frac)
        timing.append(f"{t},{-100.0 / mult:.4f},4,2,1,60,0,0")
    return "\n".join(
        [
            "osu file format v14",
            "",
            "[General]",
            "AudioFilename: audio.mp3",
            "AudioLeadIn: 0",
            "PreviewTime: 20000",
            "Countdown: 0",
            "SampleSet: Soft",
            "StackLeniency: 0.7",
            "Mode: 0",
            "",
            "[Metadata]",
            f"Title:{spec.title}",
            f"TitleUnicode:{spec.title}",
            f"Artist:{spec.artist}",
            f"ArtistUnicode:{spec.artist}",
            "Creator:osr-synth",
            f"Version:{spec.version}",
            f"Source:{spec.source}",
            f"Tags:{spec.tags}",
            "BeatmapID:0",
            "BeatmapSetID:-1",
            "",
            "[Difficulty]",
            f"HPDrainRate:{spec.hp}",
            f"CircleSize:{spec.cs}",
            f"OverallDifficulty:{spec.od}",
            f"ApproachRate:{spec.ar}",
            "SliderMultiplier:1.8",
            "SliderTickRate:1",
            "",
            "[Events]",
            "//Background and Video events",
            '0,0,"bg.jpg",0,0',
            "",
            "[TimingPoints]",
            *timing,
            "",
            "",
            "[HitObjects]",
            *b.objects,
            "",
        ]
    )


SPECS: list[Spec] = [
    Spec("crimson_overdrive", "Kagerou Signal", "Crimson Overdrive", "Technical Aim",
         "japanese rock j-rock band energetic guitar anime opening technical jumps aim",
         "", 148, 9.0, 4.0, 8.5, 6,
         [("jumps", dict(beats=16, dist=135, tech=0.45)), ("bursts", dict(count=1, size=3, dist=20, jump=110)),
          ("sliders", dict(count=6, tech=0.5)), ("rest", dict(beats=2))], seed=1,
         sv_changes=[(0.12, 1.3), (0.25, 0.8), (0.4, 1.5), (0.55, 0.7), (0.7, 1.2), (0.85, 0.9)]),
    Spec("iron_petal", "Hanabi Riot", "Iron Petal", "Aim Training",
         "j-rock japanese rock energetic fast guitar jumps aim training",
         "", 160, 9.0, 4.0, 8.5, 6,
         [("jumps", dict(beats=16, dist=144, tech=0.08)), ("bursts", dict(count=1, size=3, dist=20, jump=110)),
          ("rest", dict(beats=2))], seed=2),
    Spec("midnight_ronin", "Kagerou Signal", "Midnight Ronin", "Insane",
         "japanese rock j-rock energetic jumps",
         "", 172, 9.0, 4.0, 8, 6,
         [("jumps", dict(beats=16, dist=135, tech=0.25)), ("sliders", dict(count=6)), ("rest", dict(beats=2))],
         seed=3),
    Spec("kitsune_rush", "Hanabi Riot", "Kitsune Rush", "Deathstream",
         "j-rock japanese rock energetic streams stamina",
         "", 172, 9.2, 4.0, 8.5, 6,
         [("stream", dict(notes=32, dist=12)), ("jumps", dict(beats=4, dist=120, tech=0.1)),
          ("stream", dict(notes=24, dist=12)), ("rest", dict(beats=2))], seed=4),
    Spec("neon_katana", "Neon Shogun", "Neon Katana", "Speed",
         "japanese rock j-rock energetic streams speed",
         "", 210, 9.6, 4.2, 9, 6,
         [("stream", dict(notes=20, dist=26)), ("bursts", dict(count=3, size=5)),
          ("jumps", dict(beats=4, dist=200, tech=0.1))], seed=5),
    Spec("hyperion_gate", "Vector Lumen", "Hyperion Gate", "Extreme",
         "electronic edm trance hardstyle streams stamina long streams",
         "", 220, 9.8, 4.0, 9.5, 7,
         [("stream", dict(notes=48, dist=28)), ("rest", dict(beats=1)), ("stream", dict(notes=32, dist=30))],
         seed=6),
    Spec("thunder_shrine", "Oni no Kane", "Thunder Shrine", "Death",
         "metal japanese metal heavy aggressive deathstream stamina",
         "", 240, 9.8, 4.0, 9.5, 7,
         [("stream", dict(notes=64, dist=24)), ("jumps", dict(beats=4, dist=200, tech=0.1))], seed=7),
    Spec("sakura_drift", "Hoshino Mai", "Sakura Drift", "Hard",
         "j-pop japanese pop calm relaxing cute vocal flow",
         "", 128, 8.0, 4.0, 6, 5,
         [("flow", dict(beats=16, dist=115)), ("sliders", dict(count=8, length=110)), ("rest", dict(beats=4))],
         seed=8),
    Spec("starlit_parade", "Hoshino Mai", "Starlit Parade", "Insane",
         "j-pop japanese pop happy upbeat idol jumps",
         "", 160, 9.0, 4.0, 7.5, 5,
         [("jumps", dict(beats=16, dist=140, tech=0.15)), ("sliders", dict(count=6)), ("rest", dict(beats=2))],
         seed=9),
    Spec("glass_lullaby", "Aoi Keys", "Glass Lullaby", "Tech",
         "piano classical calm emotional slider tech sv changes",
         "", 120, 8.5, 4.0, 7, 5,
         [("sliders", dict(count=12, length=150, tech=0.8)), ("flow", dict(beats=8, dist=120))], seed=10,
         sv_changes=[(0.1, 0.6), (0.2, 1.5), (0.35, 0.75), (0.5, 1.8), (0.6, 0.5), (0.7, 1.4), (0.85, 0.9)]),
    Spec("static_bloom", "Vector Lumen", "Static Bloom", "Glitch",
         "electronic glitch hop experimental technical jumps aim",
         "", 175, 9.3, 4.2, 8.5, 6,
         [("jumps", dict(beats=16, dist=150, tech=0.65)), ("sliders", dict(count=6, tech=0.6))], seed=11,
         sv_changes=[(0.12, 1.4), (0.25, 0.7), (0.4, 1.6), (0.55, 0.6), (0.7, 1.2), (0.85, 0.8)]),
    Spec("ocean_circuit", "Miku Network", "Ocean Circuit", "Insane",
         "vocaloid electronic j-pop energetic streams medium",
         "", 180, 9.2, 4.0, 8, 6,
         [("stream", dict(notes=12, dist=28)), ("jumps", dict(beats=8, dist=150, tech=0.15)),
          ("bursts", dict(count=2, size=5))], seed=12),
    Spec("blazing_requiem", "Oni no Kane", "Blazing Requiem", "Extra",
         "metal j-metal japanese rock aggressive energetic jumps big aim",
         "", 200, 9.7, 3.8, 9, 6,
         [("jumps", dict(beats=16, dist=260, tech=0.3)), ("bursts", dict(count=2, size=3))], seed=13),
    Spec("lofi_lantern", "Quiet Harbor", "Lo-fi Lantern", "Normal",
         "lofi chill hip hop relaxing calm beginner",
         "", 90, 6.0, 3.5, 5, 4,
         [("flow", dict(beats=8, dist=110)), ("sliders", dict(count=8, length=100)), ("rest", dict(beats=4))],
         seed=14),
    Spec("paper_moon_waltz", "Aoi Keys", "Paper Moon Waltz", "Waltz",
         "classical orchestral waltz calm elegant tech rhythm",
         "", 150, 8.8, 4.2, 8, 5,
         [("jumps", dict(beats=12, dist=125, tech=0.7, div=3)), ("sliders", dict(count=6, tech=0.5))], seed=15,
         sv_changes=[(0.15, 0.75), (0.3, 1.25), (0.5, 0.6), (0.65, 1.5), (0.8, 0.9)]),
    Spec("velvet_comet", "Kirameki Club", "Velvet Comet", "Insane",
         "anime pop j-pop energetic opening mixed jumps bursts",
         "", 165, 9.2, 4.0, 8, 6,
         [("jumps", dict(beats=8, dist=160, tech=0.2)), ("bursts", dict(count=3, size=4)),
          ("stream", dict(notes=10, dist=30)), ("sliders", dict(count=4))], seed=16),
    Spec("ember_highway", "Hanabi Riot", "Ember Highway", "Hard",
         "j-rock japanese rock energetic guitar jumps easy aim",
         "", 165, 8.5, 4.0, 7, 5,
         [("jumps", dict(beats=16, dist=125, tech=0.15)), ("sliders", dict(count=6)), ("rest", dict(beats=2))],
         seed=17),
    Spec("aurora_reading", "Vector Lumen", "Aurora Signal", "AR10 Reading",
         "electronic trance high ar reading jumps",
         "", 180, 10.0, 4.5, 9, 6,
         [("jumps", dict(beats=8, dist=200, tech=0.3)), ("stream", dict(notes=8, dist=40)),
          ("jumps", dict(beats=8, dist=230, tech=0.4))], seed=18),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        name = f"{spec.artist} - {spec.title} (osr-synth) [{spec.version}].osu"
        (OUT_DIR / name).write_text(render(spec), encoding="utf-8", newline="\n")
    print(f"{len(SPECS)} beatmaps gerados em {OUT_DIR}")


if __name__ == "__main__":
    main()
