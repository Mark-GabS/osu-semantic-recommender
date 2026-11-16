"""Recomendação sobre o catálogo ONLINE do osu! (osu.ppy.sh/beatmapsets).

Pipeline por pedido:
1. `HybridQuery` (mesmo planner da busca local).
2. Catálogo: busca com os filtros nativos do osu! (`stars>4.9 stars<5.5 bpm>170 ...`),
   status ranked por padrão. Mirror sem conta ou API oficial (ver `catalog.sources`).
3. Filtro por DIFICULDADE do lado do cliente (os filtros do catálogo valem por mapset),
   exclusão de mapsets que você já tem na pasta Songs, uma dificuldade por mapset.
4. Pré-rank barato só com metadados (gênero/idioma do site, similaridade semântica,
   aderência de SR, popularidade).
5. Para os melhores candidatos: baixa o `.osu` público (com cache), roda o MESMO
   extrator mecânico da busca local e faz o rerank final (`score_map`), com o
   limite de stream aplicado sobre a contagem real de notas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..catalog import (
    CatalogSource,
    OnlineBeatmap,
    OnlineSet,
    OsuFileCache,
    detect_songs_dir,
    make_source,
    owned_set_ids,
)
from ..config import Settings, get_settings
from ..indexer.embeddings import Embedder, cosine
from ..indexer.enrichment import GENRE_KEYWORDS, Enricher, _match, build_document, template_enrichment
from ..indexer.pipeline import make_embedder
from ..indexer.vector_store import StoredMap
from ..observability import get_logger, span
from ..parser import extract_features, parse_osu_file
from .hybrid_search import MIN_STRICT_RESULTS, RANGE_FIELDS, Recommendation, expand_genres, relaxation_steps, score_map
from .query_planner import HybridQuery

log = get_logger("online")

# (genre_id, language_id) da API do osu! para um gênero canônico do planner
GENRE_TO_API: dict[str, tuple[int, int | None]] = {
    "jrock": (4, 3), "jpop": (5, 3), "kpop": (5, 6), "rock": (4, None), "pop": (5, None),
    "metal": (11, None), "anime": (3, None), "electronic": (10, None), "classical": (12, None),
    "hip hop": (9, None), "jazz": (14, None), "folk": (13, None),
}
OSU_FILTER_KEYS = {"star_rating": "stars", "bpm": "bpm", "ar": "ar", "cs": "cs", "od": "od", "length_s": "length"}


def set_genres(s: OnlineSet) -> list[str]:
    """Gêneros canônicos a partir do gênero/idioma do SITE + tags do mapset."""
    g, lang = s.genre, s.language
    out: list[str] = []
    if g == "rock":
        out += ["jrock", "rock"] if lang == "japanese" else ["rock"]
    elif g == "pop":
        out += {"japanese": ["jpop", "pop"], "korean": ["kpop", "pop"]}.get(lang, ["pop"])
    elif g == "metal":
        out += ["metal", "rock"]
    elif g not in ("unspecified", "other"):
        out.append(g)
    if lang == "instrumental":
        out.append("instrumental")
    out += _match(" ".join([s.artist, s.title, s.source, s.tags]).lower(), GENRE_KEYWORDS)
    return list(dict.fromkeys(out))


# palavra usada na busca textual quando a fonte não filtra gênero (mirror): melhora muito a
# cobertura (ex.: "classical" -> ~40% dos resultados são Classical, contra quase nada sem ela)
GENRE_KEYWORD = {
    "jrock": "rock", "rock": "rock", "metal": "metal", "jpop": "pop", "kpop": "kpop", "pop": "pop",
    "anime": "anime", "electronic": "electronic", "classical": "classical", "hip hop": "hip hop",
    "jazz": "jazz", "folk": "folk", "vocaloid": "vocaloid", "piano": "piano", "lofi": "lofi",
}
INTEGER_FILTERS = {"bpm", "length"}  # o mirror devolve vazio com decimais nesses campos


def osu_filter_string(q: HybridQuery) -> str:
    """Filtros no formato da busca do osu!. Usa >/< estritos (o mirror não entende >=);
    BPM e duração vão como inteiros. O filtro exato por dificuldade é refeito no cliente."""
    parts = []
    for name in RANGE_FIELDS:
        r = getattr(q, name)
        if r is None:
            continue
        key = OSU_FILTER_KEYS[name]
        if key in INTEGER_FILTERS:
            if r.min is not None:
                parts.append(f"{key}>{math.ceil(r.min) - 1}")
            if r.max is not None:
                parts.append(f"{key}<{math.floor(r.max) + 1}")
            continue
        if r.min is not None:
            parts.append(f"{key}>{r.min - 0.005:g}")
        if r.max is not None:
            parts.append(f"{key}<{r.max + 0.005:g}")
    return " ".join(parts)


def api_genre_filter(q: HybridQuery) -> tuple[int | None, int | None]:
    """Gênero/idioma para a API oficial, só quando o pedido aponta para um único gênero do site."""
    mapped = [GENRE_TO_API[g] for g in q.genres if g in GENRE_TO_API]
    if not mapped or len({m[0] for m in mapped}) > 1:
        return None, None
    with_lang = [m for m in mapped if m[1] is not None]
    return (with_lang or mapped)[0]


def diff_passes(b: OnlineBeatmap, q: HybridQuery) -> bool:
    if not b.is_standard:
        return False
    for name in RANGE_FIELDS:
        r = getattr(q, name)
        if r is not None and not r.contains(b.range_value(name)):
            return False
    return True


@dataclass
class OnlineRecommendation:
    rec: Recommendation
    set: OnlineSet
    beatmap: OnlineBeatmap

    @property
    def url(self) -> str:
        return self.set.beatmap_url(self.beatmap)

    @property
    def md5(self) -> str:
        return self.rec.item.features.md5


@dataclass
class OnlineResult:
    query: HybridQuery
    recommendations: list[OnlineRecommendation]
    relaxations: list[str]
    stats: dict = field(default_factory=dict)


@dataclass
class _Candidate:
    set: OnlineSet
    beatmap: OnlineBeatmap
    strict: bool
    pre_score: float = 0.0


class OnlineRecommender:
    def __init__(self, settings: Settings | None = None, source: CatalogSource | None = None,
                 files: OsuFileCache | None = None, embedder: Embedder | None = None,
                 enricher: Enricher | None = None, exclude_owned: bool = True):
        self.settings = settings or get_settings()
        self.source = source or make_source(self.settings)
        self.files = files or OsuFileCache(self.settings.osu_cache_dir, concurrency=self.settings.http_concurrency)
        self.embedder = embedder or make_embedder(self.settings)
        self.enricher = enricher
        songs = self.settings.owned_songs_dir or detect_songs_dir()
        self.owned = owned_set_ids(songs) if exclude_owned else set()

    # ------------------------------------------------------------------ etapas
    def _collect(self, q: HybridQuery, strict: bool, seen: set[int], stats: dict) -> list[_Candidate]:
        g, lang = api_genre_filter(q)
        filters = osu_filter_string(q)
        budget = self.settings.online_max_sets
        sets: list[OnlineSet] = []
        keyword = next((GENRE_KEYWORD[x] for x in q.genres if x in GENRE_KEYWORD), None)
        if keyword and not getattr(self.source, "supports_genre_filter", False):
            # mirror não filtra gênero: primeiro uma passada textual pelo gênero, depois a genérica
            sets += self.source.search(f"{keyword} {filters}".strip(), status=self.settings.online_status,
                                       max_sets=budget)
        sets += self.source.search(filters, status=self.settings.online_status, max_sets=budget,
                                   genre_id=g, language_id=lang)
        stats["sets_scanned"] = stats.get("sets_scanned", 0) + len(sets)
        center = None
        if q.star_rating and q.star_rating.min is not None and q.star_rating.max is not None:
            center = (q.star_rating.min + q.star_rating.max) / 2
        out = []
        for s in sets:
            if s.id in seen:
                continue
            if s.id in self.owned:
                stats["owned_skipped"] = stats.get("owned_skipped", 0) + 1
                seen.add(s.id)  # não recontar nos passos de relaxamento
                continue
            ok = [b for b in s.beatmaps if diff_passes(b, q)]
            if not ok:
                continue
            # uma dificuldade por mapset: a mais próxima do centro de SR pedido (ou a mais difícil)
            best = min(ok, key=lambda b: abs(b.difficulty_rating - center)) if center is not None else \
                max(ok, key=lambda b: b.difficulty_rating)
            seen.add(s.id)
            out.append(_Candidate(s, best, strict))
        return out

    def _prerank(self, cands: list[_Candidate], q: HybridQuery, q_emb: list[float]) -> None:
        if not cands:
            return
        texts = [
            f"{c.set.artist} - {c.set.title}. Source: {c.set.source}. Genre: {c.set.genre}. "
            f"Language: {c.set.language}. Genres: {', '.join(set_genres(c.set))}. Tags: {c.set.tags}"
            for c in cands
        ]
        embs = self.embedder.embed(texts)
        max_fav = max(math.log1p(c.set.favourite_count) for c in cands) or 1.0
        for c, e in zip(cands, embs):
            sem = max(0.0, cosine(q_emb, e))
            pop = math.log1p(c.set.favourite_count) / max_fav
            sr_fit = 1.0
            if q.star_rating and q.star_rating.min is not None and q.star_rating.max is not None:
                center = (q.star_rating.min + q.star_rating.max) / 2
                sr_fit = max(0.0, 1 - abs(c.beatmap.difficulty_rating - center))
            if q.genres:
                have = expand_genres(set_genres(c.set))
                music = sum(g in have for g in q.genres) / len(q.genres)
                c.pre_score = 0.45 * music + 0.30 * sem + 0.15 * sr_fit + 0.10 * pop
            else:
                c.pre_score = 0.60 * sem + 0.25 * sr_fit + 0.15 * pop

    def _analyze(self, cands: list[_Candidate], q: HybridQuery, q_emb: list[float],
                 stats: dict) -> list[OnlineRecommendation]:
        paths = self.files.get_many([(c.beatmap.id, c.beatmap.checksum) for c in cands])
        recs = []
        for c in cands:
            path = paths.get(c.beatmap.id)
            if path is None:
                continue
            try:
                f = extract_features(parse_osu_file(path))
            except Exception as exc:  # um .osu quebrado não derruba a busca
                log.debug("falha analisando %s: %s", c.beatmap.id, exc)
                continue
            # metadados oficiais do site têm prioridade sobre o que está no .osu
            f.beatmap_id, f.beatmapset_id = c.beatmap.id, c.set.id
            f.star_rating = round(c.beatmap.difficulty_rating, 2)
            e = self.enricher.enrich(f) if self.enricher else template_enrichment(f)
            e.genres = list(dict.fromkeys(set_genres(c.set) + e.genres))
            doc = build_document(f, e)
            sim = cosine(q_emb, self.embedder.embed([doc])[0])
            strict = c.strict and (q.max_stream_len is None or f.max_stream_len <= q.max_stream_len)
            rec = score_map(StoredMap(f, e, doc, distance=1 - sim), q, strict, self.settings)
            recs.append(OnlineRecommendation(rec, c.set, c.beatmap))
        stats["analyzed"] = len(recs)
        return recs

    # ------------------------------------------------------------------ API
    def search(self, q: HybridQuery) -> OnlineResult:
        stats: dict = {"source": self.source.name, "owned_sets": len(self.owned)}
        relaxations: list[str] = []
        with span("search.online", source=self.source.name, limit=q.limit) as sp:
            q_text = ". ".join([q.semantic_query] + (["Genres: " + ", ".join(q.genres)] if q.genres else [])
                               + (["Moods: " + ", ".join(q.moods)] if q.moods else []))
            q_emb = self.embedder.embed([q_text])[0]
            seen: set[int] = set()
            cands = self._collect(q, True, seen, stats)
            need = min(q.limit, MIN_STRICT_RESULTS)
            if len(cands) < need:
                for label, rq in relaxation_steps(q):
                    extra = self._collect(rq, False, seen, stats)
                    if extra:
                        relaxations.append(label)
                        cands += extra
                    if len(cands) >= need:
                        break
            stats["candidates"] = len(cands)
            self._prerank(cands, q, q_emb)
            fetch_k = min(len(cands), max(self.settings.online_fetch_k, q.limit * 2))
            top = sorted(cands, key=lambda c: (c.strict, c.pre_score), reverse=True)[:fetch_k]
            recs = self._analyze(top, q, q_emb, stats)
            recs.sort(key=lambda r: (r.rec.strict, r.rec.score), reverse=True)
            recs = recs[: q.limit]
            stats.update(osu_files=dict(self.files.stats), returned=len(recs))
            sp.set(**{k: v for k, v in stats.items() if not isinstance(v, dict)}, relaxations=relaxations)
        return OnlineResult(q, recs, relaxations, stats)
