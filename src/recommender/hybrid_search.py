"""Busca híbrida: filtros rígidos (metadados) + similaridade vetorial + rerank mecânico/musical.

Etapas:
1. `HybridQuery` -> cláusula `where` do Chroma (SR, BPM, AR, CS, OD, duração, maior stream).
2. Recuperação vetorial top-N (`candidate_pool`) dentro do subconjunto filtrado.
3. Se sobrar pouco, relaxamento progressivo e *registrado* dos filtros (os itens
   relaxados ficam abaixo dos estritos e são marcados como tal).
4. Rerank: score = w_sem·semântica + w_mec·aderência mecânica + w_mus·match de gênero/mood,
   com explicação por item (por que entrou, quanto pontuou em cada componente).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..indexer.embeddings import Embedder
from ..indexer.vector_store import StoredMap, VectorStore
from ..observability import get_logger, span
from .query_planner import HybridQuery, NumRange

log = get_logger("search")

RANGE_FIELDS = ("star_rating", "bpm", "ar", "cs", "od", "length_s")

# Hierarquia de gêneros: um subgênero implica o pai. Sem isso, um mapa rotulado só
# "jrock" (rótulo correto, comum no enriquecimento via LLM) perde pontos num pedido
# que o planner traduziu para ["jrock", "rock"].
GENRE_PARENTS: dict[str, tuple[str, ...]] = {
    "jrock": ("rock",),
    "metal": ("rock",),
    "jpop": ("pop",),
    "kpop": ("pop",),
    "vocaloid": ("jpop", "pop"),
    "hardstyle": ("electronic",),
    "trance": ("electronic",),
    "dubstep": ("electronic",),
    "piano": ("classical",),
    "orchestral": ("classical",),
}


def expand_genres(genres: list[str]) -> set[str]:
    out = {g.lower().replace("-", "").replace(" ", "") for g in genres}
    for g in list(out):
        out.update(GENRE_PARENTS.get(g, ()))
    return out
MIN_STRICT_RESULTS = 5


@dataclass
class Recommendation:
    item: StoredMap
    score: float
    semantic: float
    mechanic: float
    music: float | None
    strict: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def md5(self) -> str:
        return self.item.features.md5


@dataclass
class SearchResult:
    query: HybridQuery
    recommendations: list[Recommendation]
    where: dict | None
    relaxations: list[str]
    n_candidates: int


def _range_clauses(name: str, r: NumRange | None) -> list[dict]:
    if r is None:
        return []
    out = []
    if r.min is not None:
        out.append({name: {"$gte": float(r.min)}})
    if r.max is not None:
        out.append({name: {"$lte": float(r.max)}})
    return out


def build_where(q: HybridQuery) -> dict | None:
    clauses: list[dict] = []
    for name in RANGE_FIELDS:
        clauses += _range_clauses(name, getattr(q, name))
    if q.max_stream_len is not None:
        clauses.append({"max_stream_len": {"$lte": float(q.max_stream_len)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _widen(r: NumRange | None, delta: float) -> NumRange | None:
    if r is None:
        return None
    return NumRange(min=None if r.min is None else r.min - delta, max=None if r.max is None else r.max + delta)


def relaxation_steps(q: HybridQuery) -> list[tuple[str, HybridQuery]]:
    """Sequência de versões cada vez mais permissivas da query."""
    steps = []
    q1 = q.model_copy(deep=True)
    q1.star_rating, q1.bpm = _widen(q.star_rating, 0.4), _widen(q.bpm, 15)
    q1.ar, q1.cs, q1.od = _widen(q.ar, 0.5), _widen(q.cs, 0.5), _widen(q.od, 1.0)
    steps.append(("faixas numéricas ampliadas (SR ±0.4, BPM ±15, AR/CS ±0.5)", q1))
    q2 = q1.model_copy(deep=True)
    q2.length_s, q2.od, q2.cs = None, None, None
    q2.star_rating = _widen(q.star_rating, 0.8)
    steps.append(("removidos filtros de duração/OD/CS; SR ±0.8", q2))
    if q.max_stream_len is not None:
        q3 = q2.model_copy(deep=True)
        q3.max_stream_len = None
        steps.append(("limite de stream virou penalidade suave", q3))
    return steps


class HybridSearcher:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings | None = None):
        self.store = store
        self.embedder = embedder
        self.settings = settings or get_settings()

    def _query_text(self, q: HybridQuery) -> str:
        parts = [q.semantic_query]
        if q.genres:
            parts.append("Genres: " + ", ".join(q.genres))
        if q.moods:
            parts.append("Moods: " + ", ".join(q.moods))
        return ". ".join(parts)

    def search(self, q: HybridQuery) -> SearchResult:
        with span("search.hybrid", limit=q.limit, skills=q.skills, genres=q.genres) as sp:
            emb = self.embedder.embed([self._query_text(q)])[0]
            where = build_where(q)
            pool = max(self.settings.candidate_pool, q.limit * 5)
            strict = self.store.query(emb, where, pool)
            relaxations: list[str] = []
            relaxed: list[StoredMap] = []
            need = min(q.limit, MIN_STRICT_RESULTS)
            if len(strict) < need:
                seen = {m.features.md5 for m in strict}
                for label, rq in relaxation_steps(q):
                    extra = [m for m in self.store.query(emb, build_where(rq), pool) if m.features.md5 not in seen]
                    if extra:
                        relaxations.append(label)
                        relaxed += extra
                        seen.update(m.features.md5 for m in extra)
                    if len(strict) + len(relaxed) >= need:
                        break
            recs = [self._score(m, q, True) for m in strict] + [self._score(m, q, False) for m in relaxed]
            recs.sort(key=lambda r: (r.strict, r.score), reverse=True)
            recs = recs[: q.limit]
            sp.set(n_strict=len(strict), n_relaxed=len(relaxed), relaxations=relaxations, returned=len(recs))
        if relaxations:
            log.info("filtros relaxados: %s", "; ".join(relaxations))
        return SearchResult(q, recs, where, relaxations, len(strict) + len(relaxed))

    def _score(self, m: StoredMap, q: HybridQuery, strict: bool) -> Recommendation:
        return score_map(m, q, strict, self.settings)


# ---------------------------------------------------------------------- scoring
def score_map(m: StoredMap, q: HybridQuery, strict: bool, settings: Settings) -> Recommendation:
    """Score explicado de um mapa (compartilhado entre a busca local e a online)."""
    f, e = m.features, m.enrichment
    reasons: list[str] = []

    semantic = max(0.0, m.similarity)

    # aderência mecânica
    if q.skills:
        want = sum(f.skill_score(s) for s in q.skills) / len(q.skills)
        reasons += [f"{s} {f.skill_score(s):.2f}" for s in q.skills]
    else:
        want = 0.5
    avoid = 0.0
    if q.avoid_skills:
        avoid = max(f.skill_score(s) for s in q.avoid_skills)
        if avoid >= 0.4:
            reasons.append(f"penalidade: {', '.join(s for s in q.avoid_skills if f.skill_score(s) >= 0.4)}")
    if q.max_stream_len is not None:
        if f.max_stream_len > q.max_stream_len:
            avoid = max(avoid, min(1.0, f.max_stream_len / (3 * q.max_stream_len)))
        reasons.append(f"maior stream {f.max_stream_len}")
    sr_fit = 1.0
    if q.star_rating and q.star_rating.min is not None and q.star_rating.max is not None:
        center = (q.star_rating.min + q.star_rating.max) / 2
        half = (q.star_rating.max - q.star_rating.min) / 2 + 0.3
        sr_fit = max(0.0, 1 - abs(f.star_rating - center) / (2 * half))
    mechanic = max(0.0, 0.8 * want + 0.2 * sr_fit - 0.6 * avoid)
    reasons.insert(0, f"{f.star_rating:.2f}★ {f.bpm:.0f}BPM AR{f.ar:g}")

    # aderência musical
    music = None
    if q.genres or q.moods:
        parts, weights = [], []
        if q.genres:
            map_genres = expand_genres(e.genres)
            hit = [g for g in q.genres if g.lower().replace("-", "").replace(" ", "") in map_genres]
            parts.append(len(hit) / len(q.genres))
            weights.append(0.7)
            if hit:
                reasons.append("gênero: " + ", ".join(hit))
        if q.moods:
            hit = [x for x in q.moods if x in e.moods]
            parts.append(len(hit) / len(q.moods))
            weights.append(0.3)
            if hit:
                reasons.append("mood: " + ", ".join(hit))
        music = sum(p * w for p, w in zip(parts, weights)) / sum(weights)

    ws, wm = settings.semantic_weight, settings.mechanic_weight
    wmus = settings.music_weight if music is not None else 0.0
    total = ws + wm + wmus
    score = (ws * semantic + wm * mechanic + wmus * (music or 0.0)) / total
    if not strict:
        reasons.append("fora dos filtros estritos (relaxado)")
    return Recommendation(m, round(score, 4), round(semantic, 4), round(mechanic, 4),
                          None if music is None else round(music, 4), strict, reasons)
