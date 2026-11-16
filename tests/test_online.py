"""Busca online com catálogo e arquivos simulados (sem rede)."""

import hashlib
import shutil

import pytest

from src.catalog import OnlineSet, owned_set_ids
from src.indexer import HashEmbedder
from src.recommender import HeuristicQueryPlanner, HybridQuery, NumRange
from src.recommender.online_search import (
    OnlineRecommender,
    api_genre_filter,
    diff_passes,
    osu_filter_string,
    set_genres,
)

PROMPT = "quero mapas de 5.2★ com jumps técnicos, rock japonês energético e poucas streams longas para treinar aim"


def _set(set_id, title, genre, lang, diffs, artist="Artist", tags="", favs=10):
    return OnlineSet.from_api({
        "id": set_id, "artist": artist, "title": title, "tags": tags, "status": "ranked",
        "genre": {"id": genre, "name": "x"}, "language": {"id": lang, "name": "y"}, "favourite_count": favs,
        "beatmaps": [{"id": bid, "beatmapset_id": set_id, "version": f"d{bid}", "mode": mode,
                      "difficulty_rating": sr, "bpm": 180, "ar": 9, "cs": 4, "accuracy": 8, "drain": 5,
                      "total_length": 120, "checksum": md5} for bid, sr, md5, mode in diffs],
    })


class FakeSource:
    name = "fake"

    def __init__(self, sets):
        self.sets, self.queries = sets, []

    def search(self, query, *, status, max_sets, genre_id=None, language_id=None):
        self.queries.append(query)
        return self.sets[:max_sets]


class FakeFiles:
    """Serve .osu locais (mapas sintéticos) no lugar do endpoint osu.ppy.sh/osu/{id}."""

    def __init__(self, mapping):
        self.mapping, self.stats = mapping, {"downloaded": 0}

    def get_many(self, items):
        return {bid: self.mapping.get(bid) for bid, _ in items}


def by_title(sample_files, title):
    return next(p for p in sample_files if f"- {title} (" in p.name)


@pytest.fixture
def catalog(sample_files):
    crimson, kitsune, glass = (by_title(sample_files, t) for t in ("Crimson Overdrive", "Kitsune Rush", "Glass Lullaby"))
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    sets = [
        _set(100, "Crimson Overdrive", 4, 3, [(1, 5.40, md5(crimson), "osu"), (2, 3.1, "a" * 32, "osu")],
             artist="Kagerou Signal", tags="j-rock energetic"),
        # j-rock no SR certo, mas o .osu revela streams longas (o catálogo não sabe disso)
        _set(200, "Kitsune Rush", 4, 3, [(3, 5.30, md5(kitsune), "osu")], artist="Hanabi Riot"),
        # SR certo, gênero errado
        _set(300, "Glass Lullaby", 12, 5, [(4, 5.20, md5(glass), "osu")], artist="Aoi Keys"),
        # só a dificuldade de taiko passa no SR: deve ser ignorado
        _set(400, "Taiko Only", 4, 3, [(5, 5.2, "b" * 32, "taiko"), (6, 2.0, "c" * 32, "osu")]),
    ]
    files = FakeFiles({1: crimson, 3: kitsune, 4: glass})
    return sets, files


def make(settings, sets, files, owned=()):
    rec = OnlineRecommender(settings, source=FakeSource(sets), files=files,
                            embedder=HashEmbedder(settings.hash_dim), exclude_owned=False)
    rec.owned = set(owned)
    return rec


def test_filter_string_uses_strict_operators():
    q = HybridQuery(semantic_query="x", collection_name="c", star_rating=NumRange(min=4.9, max=5.5),
                    bpm=NumRange(min=170))
    assert osu_filter_string(q) == "stars>4.895 stars<5.505 bpm>169"
    q.bpm = NumRange(min=175, max=185)
    assert osu_filter_string(q).endswith("bpm>174 bpm<186")  # inteiros: o mirror falha com decimais


def test_genre_mapping_from_site_metadata():
    assert set_genres(_set(1, "t", 4, 3, [])) [:2] == ["jrock", "rock"]
    assert set_genres(_set(1, "t", 11, 2, []))[:2] == ["metal", "rock"]
    assert "jpop" in set_genres(_set(1, "t", 5, 3, []))
    assert api_genre_filter(HeuristicQueryPlanner().plan(PROMPT)) == (4, 3)


def test_diff_filter_is_per_difficulty_and_standard_only(catalog):
    sets, _ = catalog
    q = HeuristicQueryPlanner().plan(PROMPT)
    assert [b.id for b in sets[0].beatmaps if diff_passes(b, q)] == [1]
    assert not any(diff_passes(b, q) for b in sets[3].beatmaps)


def test_online_pipeline_reranks_with_real_mechanics(settings, catalog):
    sets, files = catalog
    res = make(settings, sets, files).search(HeuristicQueryPlanner().plan(PROMPT))
    titles = [r.set.title for r in res.recommendations]
    assert titles[0] == "Crimson Overdrive"
    assert "Taiko Only" not in titles
    kitsune = next(r for r in res.recommendations if r.set.title == "Kitsune Rush")
    assert not kitsune.rec.strict  # maior stream real (32) > limite 12
    top = res.recommendations[0]
    assert top.url == "https://osu.ppy.sh/beatmapsets/100#osu/1"
    assert top.md5 == sets[0].beatmaps[0].checksum  # collection.db usa o MD5 do .osu
    assert top.rec.item.features.star_rating == 5.40  # SR oficial do site prevalece


def test_mirror_gets_extra_genre_keyword_pass(settings, catalog):
    sets, files = catalog
    rec = make(settings, sets, files)
    rec.search(HeuristicQueryPlanner().plan(PROMPT))
    assert rec.source.queries[0].startswith("rock stars>")  # passada textual pelo gênero
    assert rec.source.queries[1].startswith("stars>")       # passada genérica


def test_owned_sets_are_hidden(settings, catalog):
    sets, files = catalog
    res = make(settings, sets, files, owned={100}).search(HeuristicQueryPlanner().plan(PROMPT))
    assert "Crimson Overdrive" not in [r.set.title for r in res.recommendations]
    assert res.stats["owned_skipped"] == 1


def test_owned_set_ids_reads_songs_folder_names(tmp_path):
    for name in ("123456 Artist - Title", "99 X - Y", "sem numero", "777 arquivo.osz"):
        (tmp_path / name).mkdir() if not name.endswith(".osz") else (tmp_path / name).write_text("")
    assert owned_set_ids(tmp_path) == {123456, 99}
    assert owned_set_ids(None) == set()


def test_osu_file_cache_validates_and_caches(tmp_path, sample_files):
    import httpx

    from src.catalog import OsuFileCache

    body = sample_files[0].read_bytes()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path.endswith("/999"):
            return httpx.Response(200, content=b"<html>not found</html>")
        return httpx.Response(200, content=body)

    cache = OsuFileCache(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    md5 = hashlib.md5(body).hexdigest()
    assert cache.get(42, md5).read_bytes() == body
    assert cache.get(42, md5) is not None and len(calls) == 1  # 2ª vez vem do cache
    assert cache.get(999) is None  # resposta que não é .osu é rejeitada
    shutil.rmtree(tmp_path)
