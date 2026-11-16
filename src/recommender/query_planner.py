"""Query planner: linguagem natural -> `HybridQuery` (Structured Outputs).

Dois planners com o MESMO contrato de saída:
- `LLMQueryPlanner`: Claude via `messages.parse(output_format=HybridQuery)`;
  o schema Pydantic é imposto pela API, então o resultado é sempre válido.
- `HeuristicQueryPlanner`: regras/regex PT+EN, determinístico e offline.
  Serve como fallback (sem chave, erro de API, recusa) e como baseline nos evals.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..indexer.embeddings import normalize_tokens
from ..llm import LLMClient, LLMError
from ..observability import get_logger, span

log = get_logger("planner")

Skill = Literal[
    "jumps", "tech", "streams", "long_streams", "bursts", "stamina", "flow_aim", "high_ar", "sliders", "speed"
]


class NumRange(BaseModel):
    min: float | None = None
    max: float | None = None

    def contains(self, v: float) -> bool:
        return (self.min is None or v >= self.min) and (self.max is None or v <= self.max)


class HybridQuery(BaseModel):
    """Consulta híbrida: filtros rígidos numéricos + sinais semânticos + preferências mecânicas."""

    star_rating: NumRange | None = Field(default=None, description="faixa de star rating")
    bpm: NumRange | None = None
    ar: NumRange | None = None
    cs: NumRange | None = None
    od: NumRange | None = None
    length_s: NumRange | None = Field(default=None, description="duração do mapa em segundos")
    max_stream_len: int | None = Field(
        default=None, description="limite rígido para a maior stream (notas 1/4 consecutivas)"
    )
    skills: list[Skill] = Field(default_factory=list, description="padrões mecânicos desejados (boost no rerank)")
    avoid_skills: list[Skill] = Field(default_factory=list, description="padrões a evitar (penalidade no rerank)")
    genres: list[str] = Field(default_factory=list, description="gêneros canônicos em inglês minúsculo")
    moods: list[str] = Field(default_factory=list, description="moods canônicos em inglês minúsculo")
    semantic_query: str = Field(description="reescrita em inglês para busca vetorial (música + estilo)")
    training_goal: str | None = Field(default=None, description="ex: 'aim', 'speed', 'stamina', 'reading'")
    limit: int = Field(default=20, ge=1, le=200)
    collection_name: str = Field(description="nome curto para a coleção no osu!")
    rationale: str = Field(default="", description="1 frase explicando as decisões de filtro")
    planner: str = Field(default="heuristic", description="preenchido pelo sistema")


SYSTEM_PROMPT = """You translate osu!standard beatmap requests (Portuguese or English) into a HybridQuery.

Conventions (follow strictly, they are evaluated):
- Star rating: a single value "X★"/"X stars"/"X estrelas" -> min X-0.3, max X+0.3.
  "around/~/por volta de X" -> ±0.3. "até/up to/below X" -> max X. "acima de/above/X+" -> min X.
  "entre X e Y"/"X-Y" -> [X, Y]. Missing -> null.
- BPM, AR, CS, OD: same conventions but exact single values become ±5 BPM / ±0.3 for AR/CS/OD.
- length_s: "curtos/short" -> max 120; "longos/long maps" (the map, not streams) -> min 180; "marathon" -> min 300.
- Streams: "poucas streams longas"/"few long streams" -> max_stream_len 12 and avoid_skills
  [long_streams, stamina]. "sem streams"/"no streams" -> max_stream_len 6 and avoid
  [streams, long_streams]. "deathstream"/"streams longas" desired -> skills [long_streams, stamina].
- Mechanics vocabulary: jumps/aim -> "jumps"; técnico/technical/tech -> "tech"; "flow" -> "flow_aim";
  bursts -> "bursts"; "AR alto"/"high AR"/reading -> "high_ar"; speed/velocidade -> "speed";
  resistência/stamina -> "stamina"; sliders/SV -> "sliders".
- "para treinar aim" -> training_goal "aim" and include "jumps" in skills.
- genres/moods: lowercase English canonical labels (jrock, rock, metal, jpop, anime, vocaloid,
  electronic, classical, piano, lofi; energetic, calm, aggressive, happy, emotional, elegant).
  "rock japonês" -> genres [jrock, rock].
- semantic_query: one English sentence combining music (genre, mood) and play style, for vector search.
- limit: number of maps requested, default 20.
- collection_name: short (<= 40 chars), e.g. "5.2* tech jumps jrock".
- Never invent constraints the user did not ask for."""


class LLMQueryPlanner:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)

    def plan(self, prompt: str) -> HybridQuery:
        with span("planner.llm", backend=self.llm.backend_name, model=self.settings.planner_model or "default"):
            q = self.llm.structured(
                model=self.settings.planner_model,
                system=SYSTEM_PROMPT,
                user=f"Request: {prompt}",
                schema=HybridQuery,
                purpose="query_planner",
                effort="medium",
            )
        q.planner = self.llm.label(self.settings.planner_model)
        return q


# ---------------------------------------------------------------------------
# Planner heurístico
# ---------------------------------------------------------------------------
_NUM = r"(\d+(?:[.,]\d+)?)"
_STAR = r"(?:★|\*|stars?|estrelas?|sr\b)"
_NEG = {"poucas", "pouca", "poucos", "pouco", "sem", "evitar", "evite", "nada", "nenhuma", "nenhum",
        "no", "without", "few", "avoid", "less", "menos", "not", "nao"}
_GENRES = ["jrock", "rock", "metal", "jpop", "anime", "vocaloid", "electronic", "classical", "piano", "lofi"]
_MOODS = ["energetic", "calm", "aggressive", "happy", "emotional", "elegant", "relaxing", "intense", "sad"]
_MOOD_ALIASES = {"relaxing": "calm", "intense": "aggressive", "sad": "emotional"}

# token canônico -> skill
_SKILL_TOKENS = {
    "jumps": "jumps", "aim": "jumps", "technical": "tech", "flow": "flow_aim", "bursts": "bursts",
    "stamina": "stamina", "speed": "speed", "reading": "high_ar", "sliders": "sliders", "slider": "sliders",
    "sv": "sliders",
    "deathstream": "long_streams", "streams": "streams",
}


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def _range_for(text: str, unit: str, tol: float) -> NumRange | None:
    t = text
    m = re.search(rf"(?:entre|between)\s*{_NUM}\s*(?:e|and|a|-)\s*{_NUM}\s*{unit}", t)
    if m:
        return NumRange(min=_f(m[1]), max=_f(m[2]))
    m = re.search(rf"{_NUM}\s*(?:-|–|a|to)\s*{_NUM}\s*{unit}", t)
    if m:
        return NumRange(min=_f(m[1]), max=_f(m[2]))
    m = re.search(rf"(?:at[eé]|up to|below|abaixo de|menos de|under|max(?:imo)?)\s*{_NUM}\s*{unit}", t)
    if m:
        return NumRange(max=_f(m[1]))
    m = re.search(rf"(?:acima de|above|mais de|over|min(?:imo)?|pelo menos|at least)\s*{_NUM}\s*{unit}", t)
    if m:
        return NumRange(min=_f(m[1]))
    m = re.search(rf"{_NUM}\s*\+\s*{unit}|{_NUM}\s*{unit}\s*\+", t)
    if m:
        return NumRange(min=_f(m[1] or m[2]))
    m = re.search(rf"{_NUM}\s*{unit}", t)
    if m:
        v = _f(m[1])
        return NumRange(min=round(v - tol, 2), max=round(v + tol, 2))
    return None


def _prefixed(text: str, key: str, tol: float) -> NumRange | None:
    m = re.search(rf"\b{key}\s*(?:>=|>|acima de|above)\s*{_NUM}", text)
    if m:
        return NumRange(min=_f(m[1]))
    m = re.search(rf"\b{key}\s*(?:<=|<|abaixo de|below|at[eé])\s*{_NUM}", text)
    if m:
        return NumRange(max=_f(m[1]))
    m = re.search(rf"\b{key}\s*(?:alto|high)\b|\b(?:alto|high)\s*{key}\b", text)
    if m and key == "ar":
        return NumRange(min=9.6)
    m = re.search(rf"\b{key}\s*=?\s*{_NUM}\b", text)
    if m:
        v = _f(m[1])
        return NumRange(min=round(v - tol, 2), max=round(v + tol, 2))
    return None


def _negated(tokens: list[str], idx: int, window: int = 3) -> bool:
    return any(t in _NEG for t in tokens[max(0, idx - window):idx])


class HeuristicQueryPlanner:
    def plan(self, prompt: str) -> HybridQuery:
        with span("planner.heuristic"):
            return self._plan(prompt)

    def _plan(self, prompt: str) -> HybridQuery:
        raw = prompt.lower()
        text = " ".join(normalize_tokens(prompt)) + " " + raw  # tokens canônicos + texto original
        toks = normalize_tokens(prompt)

        sr = _range_for(raw, _STAR, 0.3)
        bpm = _range_for(raw, r"bpm", 5) or _prefixed(raw, "bpm", 5)
        ar = _prefixed(raw, "ar", 0.3)
        cs = _prefixed(raw, "cs", 0.3)
        od = _prefixed(raw, "od", 0.3)

        length = None
        if re.search(r"\b(mapas?|maps?|m[uú]sicas?|songs?)\s+(curt[oa]s?|short)\b|\bshort maps?\b", raw):
            length = NumRange(max=120)
        elif re.search(r"\bmarathon|maratona", raw):
            length = NumRange(min=300)
        elif re.search(r"\b(mapas?|maps?|m[uú]sicas?|songs?)\s+(long[oa]s?|long)\b|\blong maps?\b", raw):
            length = NumRange(min=180)

        skills: list[str] = []
        avoid: list[str] = []
        max_stream: int | None = None

        def add(lst: list[str], *items: str) -> None:
            for it in items:
                if it not in lst:
                    lst.append(it)

        for i, tok in enumerate(toks):
            skill = _SKILL_TOKENS.get(tok)
            if not skill:
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            prev = toks[i - 1] if i else ""
            is_long = nxt in {"longas", "longa", "long", "longos"} or prev in {"long", "longas", "longa"}
            neg = _negated(toks, i)
            if skill == "streams":
                if neg and is_long:
                    max_stream = 12
                    add(avoid, "long_streams", "stamina")
                elif neg:
                    max_stream = 6
                    add(avoid, "streams", "long_streams")
                elif is_long:
                    add(skills, "long_streams", "stamina")
                else:
                    add(skills, "streams")
            elif neg:
                add(avoid, skill)
            else:
                add(skills, skill)

        if "high_ar" not in skills and ar and ar.min is not None and ar.min >= 9.6:
            add(skills, "high_ar")
        training = None
        m = re.search(r"(?:treinar|training|train|praticar|practice)\s+(?:de\s+|meu\s+|my\s+)?(\w+)", raw)
        if m:
            goal = normalize_tokens(m[1])[0] if normalize_tokens(m[1]) else m[1]
            training = {"jumps": "aim", "stamina": "stamina", "speed": "speed", "reading": "reading",
                        "technical": "tech", "streams": "speed"}.get(goal, goal)
            if training == "aim":
                add(skills, "jumps")
            elif training == "stamina":
                add(skills, "stamina")
            elif training == "reading":
                add(skills, "high_ar")
        if "tech" in skills and "jumps" not in skills and "jumps" in text:
            add(skills, "jumps")

        genres = [g for g in _GENRES if g in toks]
        if "jrock" in genres and "rock" not in genres:
            genres.append("rock")
        if "japanese" in toks and "rock" in toks and "jrock" not in genres:
            genres = ["jrock", *genres]
        if "metal" in genres and "rock" not in genres:
            pass
        moods = []
        for mtok in _MOODS:
            if mtok in toks:
                canon = _MOOD_ALIASES.get(mtok, mtok)
                if canon not in moods:
                    moods.append(canon)

        limit = 20
        m = re.search(r"\b(\d{1,3})\s*(?:mapas|maps|beatmaps|m[uú]sicas|songs)\b", raw)
        if m:
            limit = max(1, min(200, int(m[1])))

        style_words = {
            "jumps": "jumps", "tech": "technical", "streams": "streams", "long_streams": "long streams",
            "bursts": "bursts", "stamina": "stamina", "flow_aim": "flow aim", "high_ar": "high ar reading",
            "sliders": "slider tech", "speed": "speed",
        }
        semantic = " ".join(
            [*moods, *genres, "music", *(style_words[s] for s in skills)]
            + (["no long streams"] if "long_streams" in avoid else [])
        ).strip()
        sr_label = f"{(sr.min + sr.max) / 2:.1f}*" if sr and sr.min is not None and sr.max is not None else ""
        name = " ".join(x for x in [sr_label, *[style_words[s] for s in skills[:2]], *genres[:1]] if x)
        return HybridQuery(
            star_rating=sr, bpm=bpm, ar=ar, cs=cs, od=od, length_s=length,
            max_stream_len=max_stream, skills=skills, avoid_skills=avoid,
            genres=genres, moods=moods, semantic_query=semantic or prompt,
            training_goal=training, limit=limit,
            collection_name=("osr: " + (name or "recommendations"))[:40],
            rationale="heuristic regex/keyword planner", planner="heuristic",
        )


class QueryPlanner:
    """Fachada: tenta LLM, cai para heurística em qualquer falha (com log)."""

    def __init__(self, settings: Settings | None = None, use_llm: bool = True, llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.heuristic = HeuristicQueryPlanner()
        self.llm_planner = LLMQueryPlanner(self.settings, llm) if use_llm else None

    def plan(self, prompt: str) -> HybridQuery:
        if self.llm_planner and self.llm_planner.llm.available:
            try:
                return self.llm_planner.plan(prompt)
            except LLMError as exc:
                log.warning("planner LLM falhou (%s); usando heurístico", exc)
        return self.heuristic.plan(prompt)
