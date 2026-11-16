"""Testes de planner, busca híbrida e do contrato com o LLM (cliente mockado, sem rede)."""

from types import SimpleNamespace

import pytest

from src.indexer import Enricher, HashEmbedder, VectorStore, index_songs
from src.indexer.enrichment import MapEnrichment
from src.llm import LLMClient
from src.recommender import HeuristicQueryPlanner, HybridQuery, HybridSearcher, NumRange, QueryPlanner, build_where

PROMPT = "quero mapas de 5.2★ com jumps técnicos, rock japonês energético e poucas streams longas para treinar aim"


def test_heuristic_planner_main_prompt():
    q = HeuristicQueryPlanner().plan(PROMPT)
    assert q.star_rating == NumRange(min=4.9, max=5.5)
    assert {"jumps", "tech"} <= set(q.skills)
    assert "long_streams" in q.avoid_skills and q.max_stream_len == 12
    assert "jrock" in q.genres and "energetic" in q.moods
    assert q.training_goal == "aim"


@pytest.mark.parametrize("prompt,check", [
    ("mapas entre 4 e 5 estrelas", lambda q: q.star_rating == NumRange(min=4, max=5)),
    ("maps up to 3.5 stars", lambda q: q.star_rating == NumRange(max=3.5)),
    ("acima de 7 estrelas", lambda q: q.star_rating == NumRange(min=7)),
    ("200 bpm streams", lambda q: q.bpm == NumRange(min=195, max=205) and "streams" in q.skills),
    ("sem streams, AR alto", lambda q: q.max_stream_len == 6 and q.ar.min >= 9.6),
    ("15 mapas curtos", lambda q: q.limit == 15 and q.length_s.max == 120),
])
def test_heuristic_planner_conventions(prompt, check):
    assert check(HeuristicQueryPlanner().plan(prompt))


def test_build_where():
    q = HybridQuery(semantic_query="x", collection_name="c", star_rating=NumRange(min=5, max=6), max_stream_len=12)
    assert build_where(q) == {"$and": [{"star_rating": {"$gte": 5.0}}, {"star_rating": {"$lte": 6.0}},
                                       {"max_stream_len": {"$lte": 12.0}}]}
    assert build_where(HybridQuery(semantic_query="x", collection_name="c")) is None


@pytest.fixture
def indexed(settings):
    emb = HashEmbedder(settings.hash_dim)
    store = VectorStore(settings, embedder_name=emb.name, ephemeral=True)
    report = index_songs(settings=settings, use_llm=False, store=store, embedder=emb)
    assert report.indexed == 18 and not report.errors
    return settings, store, emb


def test_incremental_indexing_skips_existing(indexed):
    settings, store, emb = indexed
    again = index_songs(settings=settings, use_llm=False, store=store, embedder=emb)
    assert again.indexed == 0 and again.skipped_existing == 18


def test_end_to_end_main_prompt(indexed):
    settings, store, emb = indexed
    q = HeuristicQueryPlanner().plan(PROMPT)
    res = HybridSearcher(store, emb, settings).search(q)
    top = res.recommendations[0].item.features
    assert top.title == "Crimson Overdrive"
    strict = [r for r in res.recommendations if r.strict]
    assert strict and all(4.9 <= r.item.features.star_rating <= 5.5 for r in strict)
    assert all(r.item.features.max_stream_len <= 12 for r in strict)
    titles = [r.item.features.title for r in res.recommendations]
    assert "Kitsune Rush" not in titles[:3]  # j-rock, SR próximo, mas streams longas


def test_relaxation_is_reported(indexed):
    settings, store, emb = indexed
    q = HybridQuery(semantic_query="metal", collection_name="c", star_rating=NumRange(min=8.6, max=9.0), limit=5)
    res = HybridSearcher(store, emb, settings).search(q)
    assert res.relaxations and all(not r.strict for r in res.recommendations)


# ---------------------------------------------------------------------- LLM mockado
class FakeMessages:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed, self.stop_reason, self.calls = parsed, stop_reason, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason, model=kwargs["model"],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0))


def fake_client(parsed, stop_reason="end_turn"):
    msgs = FakeMessages(parsed, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_llm_planner_uses_structured_outputs(settings):
    expected = HybridQuery(semantic_query="energetic jrock technical jumps", collection_name="x",
                           star_rating=NumRange(min=4.9, max=5.5), skills=["jumps", "tech"])
    client, msgs = fake_client(expected)
    q = QueryPlanner(settings, llm=LLMClient(settings, client=client)).plan(PROMPT)
    assert q.star_rating == expected.star_rating and q.planner.startswith("llm:")
    call = msgs.calls[0]
    assert call["output_format"] is HybridQuery
    assert call["model"] == "claude-opus-5-5"  # vazio -> modelo padrão da API
    assert call["fallbacks"] == "default"


def test_planner_falls_back_on_refusal(settings):
    client, _ = fake_client(None, stop_reason="refusal")
    q = QueryPlanner(settings, llm=LLMClient(settings, client=client)).plan(PROMPT)
    assert q.planner == "heuristic" and q.star_rating == NumRange(min=4.9, max=5.5)


def test_enricher_llm_result_is_cached(settings, sample_files):
    from src.parser import extract_features, parse_osu_file

    f = extract_features(parse_osu_file(sample_files[0]))
    enr = MapEnrichment(summary="s", genres=["jrock"], moods=["energetic"], play_style=["jumps"],
                        training_focus=["aim"], energy="high")
    client, msgs = fake_client(enr)
    e1 = Enricher(settings, llm=LLMClient(settings, client=client))
    out = e1.enrich(f)
    assert out.source == "llm:anthropic_api:default" and len(msgs.calls) == 1
    e1.save_cache()
    e2 = Enricher(settings, llm=LLMClient(settings, client=client))
    e2.enrich(f)
    assert len(msgs.calls) == 1 and e2.stats["cache_hits"] == 1


def test_genre_hierarchy_counts_subgenre_as_parent():
    from src.recommender.hybrid_search import expand_genres

    assert expand_genres(["jrock"]) == {"jrock", "rock"}
    assert expand_genres(["J-Rock"]) == {"jrock", "rock"}
    assert "pop" in expand_genres(["vocaloid"]) and "classical" in expand_genres(["piano"])


def test_duplicate_osu_files_are_indexed_once(settings, sample_files, tmp_path):
    """Mesmo .osu em duas pastas da Songs (mapset baixado 2x) não pode quebrar o upsert."""
    import shutil

    songs = tmp_path / "Songs"
    for folder in ("123 Set", "123 Set (1)"):
        (songs / folder).mkdir(parents=True)
        for p in sample_files[:3]:
            shutil.copy(p, songs / folder / p.name)
    emb = HashEmbedder(settings.hash_dim)
    store = VectorStore(settings, embedder_name=emb.name, ephemeral=True)
    report = index_songs(songs, settings=settings, use_llm=False, store=store, embedder=emb)
    assert report.indexed == 3 and report.skipped_duplicates == 3 and not report.errors
    assert store.count() == 3
