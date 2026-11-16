"""Pipeline de indexação: .osu -> parse -> features -> enrichment -> embedding -> Chroma.

Incremental: beatmaps cujo MD5 já está no índice são pulados (a menos de `reset`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings, get_settings
from ..observability import get_logger, span
from ..parser import OsuParseError, extract_features, iter_osu_files, parse_osu_file
from ..parser.feature_extractor import MapFeatures
from .embeddings import Embedder, get_embedder
from .enrichment import Enricher, build_document
from .vector_store import VectorStore

log = get_logger("indexer")


@dataclass
class IndexReport:
    scanned: int = 0
    indexed: int = 0
    skipped_existing: int = 0
    skipped_mode: int = 0
    skipped_duplicates: int = 0
    errors: list[str] = field(default_factory=list)
    enrichment_stats: dict = field(default_factory=dict)


def make_embedder(settings: Settings) -> Embedder:
    return get_embedder(settings.embedding_backend, dim=settings.hash_dim, model_name=settings.fastembed_model)


def index_songs(
    songs_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    use_llm: bool = True,
    reset: bool = False,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    settings = settings or get_settings()
    songs_dir = Path(songs_dir or settings.songs_dir)
    embedder = embedder or make_embedder(settings)
    store = store or VectorStore(settings, embedder_name=embedder.name)
    if reset:
        store.reset()
    report = IndexReport()
    enricher = Enricher(settings, use_llm=use_llm)

    with span("index.run", songs_dir=str(songs_dir), embedder=embedder.name, llm=enricher.use_llm) as sp:
        existing = store.existing_ids()
        files = list(iter_osu_files(songs_dir))
        feats: list[MapFeatures] = []
        seen: set[str] = set()
        for i, path in enumerate(files, 1):
            report.scanned += 1
            try:
                bm = parse_osu_file(path)
            except (OsuParseError, OSError) as exc:
                report.errors.append(f"{path.name}: {exc}")
                continue
            if bm.mode != 0:
                report.skipped_mode += 1
                continue
            if bm.md5 in existing:
                report.skipped_existing += 1
                continue
            if bm.md5 in seen:
                # mesmo .osu em duas pastas (mapset baixado duas vezes): o MD5 é o ID,
                # e o osu! também trata os dois como o mesmo beatmap
                report.skipped_duplicates += 1
                continue
            seen.add(bm.md5)
            try:
                feats.append(extract_features(bm))
            except Exception as exc:  # um mapa quebrado não derruba a indexação
                report.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            if progress:
                progress(i, len(files) * 2)

        # enriquecimento em paralelo: cada chamada LLM (subprocesso do Claude Code ou
        # requisição HTTP) é I/O-bound, então threads bastam
        workers = max(1, settings.llm_concurrency) if enricher.use_llm else 1
        enrichments: list = [None] * len(feats)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(enricher.enrich, f): idx for idx, f in enumerate(feats)}
            for j, fut in enumerate(as_completed(futures), 1):
                enrichments[futures[fut]] = fut.result()
                if progress:
                    progress(len(files) + int(j / max(1, len(feats)) * len(files)), len(files) * 2)
                if j % 50 == 0:
                    enricher.save_cache()
        enricher.save_cache()
        items = [(f, e, build_document(f, e)) for f, e in zip(feats, enrichments)]

        if items:
            vectors = embedder.embed([doc for *_, doc in items])
            store.upsert([(f, e, doc, v) for (f, e, doc), v in zip(items, vectors)])
        report.indexed = len(items)
        report.enrichment_stats = enricher.stats
        sp.set(scanned=report.scanned, indexed=report.indexed, errors=len(report.errors), **enricher.stats)
    log.info("indexação: %d novos, %d já existentes, %d erros", report.indexed, report.skipped_existing,
             len(report.errors))
    return report
