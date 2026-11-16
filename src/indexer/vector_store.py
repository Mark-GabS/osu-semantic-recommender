"""Banco vetorial local (ChromaDB) com metadados estruturados para filtros rígidos.

Cada beatmap vira um registro com:
- `embedding`: vetor do documento semântico (enriquecimento + metadados);
- `metadata`: features numéricas achatadas (filtráveis via `where` do Chroma:
  `star_rating`, `bpm`, `ar`, `cs`, `max_stream_len`...), flags booleanas
  `skill_<x>` e o JSON completo de features/enriquecimento para reconstrução.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import Settings, get_settings
from ..parser.feature_extractor import SKILLS, MapFeatures
from .enrichment import MapEnrichment

FILTERABLE = (
    "star_rating", "bpm", "ar", "cs", "od", "hp", "length_s", "max_stream_len", "n_long_streams",
    "n_streams", "n_bursts", "jump_ratio", "aim_score", "speed_score", "stamina_score", "tech_score",
    "flow_score", "reading_score", "slider_ratio", "notes_per_second",
)


class EmbedderMismatch(RuntimeError):
    pass


@dataclass
class StoredMap:
    features: MapFeatures
    enrichment: MapEnrichment
    document: str
    distance: float | None = None

    @property
    def similarity(self) -> float:
        # distância cosseno do Chroma: d = 1 - cos
        return 1.0 - self.distance if self.distance is not None else 0.0


def _metadata(f: MapFeatures, e: MapEnrichment) -> dict[str, Any]:
    data = f.model_dump()
    md: dict[str, Any] = {k: float(data[k]) for k in FILTERABLE}
    md.update(
        md5=f.md5,
        title=f.title,
        artist=f.artist,
        version=f.version,
        mode=f.mode,
        genres="|".join(e.genres),
        enrichment_source=e.source,
        features_json=f.model_dump_json(),
        enrichment_json=e.model_dump_json(),
    )
    for s in SKILLS:
        md[f"skill_{s}"] = s in f.skills
    return md


def _from_record(meta: dict[str, Any], doc: str, distance: float | None = None) -> StoredMap:
    return StoredMap(
        features=MapFeatures.model_validate_json(meta["features_json"]),
        enrichment=MapEnrichment.model_validate_json(meta["enrichment_json"]),
        document=doc,
        distance=distance,
    )


class VectorStore:
    def __init__(self, settings: Settings | None = None, embedder_name: str = "hash-512", ephemeral: bool = False):
        self.settings = settings or get_settings()
        chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        if ephemeral:
            self.client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.settings.chroma_dir), settings=chroma_settings)
        self.embedder_name = embedder_name
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine", "embedder": embedder_name},
            embedding_function=None,
        )
        stored = (self.collection.metadata or {}).get("embedder")
        if stored and stored != embedder_name and self.collection.count() > 0:
            raise EmbedderMismatch(
                f"índice criado com '{stored}', mas o embedder atual é '{embedder_name}'. "
                "Rode `osr index --reset` para reindexar."
            )

    def reset(self) -> None:
        self.client.delete_collection(self.settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine", "embedder": self.embedder_name},
            embedding_function=None,
        )

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def upsert(self, items: list[tuple[MapFeatures, MapEnrichment, str, list[float]]], batch: int = 256) -> None:
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            self.collection.upsert(
                ids=[f.md5 for f, *_ in chunk],
                embeddings=[emb for *_, emb in chunk],
                documents=[doc for _, _, doc, _ in chunk],
                metadatas=[_metadata(f, e) for f, e, _, _ in chunk],
            )

    def query(self, embedding: list[float], where: dict | None, n: int) -> list[StoredMap]:
        total = self.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            where=where or None,
            include=["metadatas", "documents", "distances"],
        )
        metas, docs, dists = res["metadatas"][0], res["documents"][0], res["distances"][0]
        return [_from_record(m, d, dist) for m, d, dist in zip(metas, docs, dists)]

    def get(self, md5s: list[str] | None = None, where: dict | None = None) -> list[StoredMap]:
        res = self.collection.get(ids=md5s, where=where or None, include=["metadatas", "documents"])
        return [_from_record(m, d) for m, d in zip(res["metadatas"], res["documents"])]

    def all(self) -> list[StoredMap]:
        return self.get()


def as_json(sm: StoredMap) -> str:
    return json.dumps({"features": sm.features.model_dump(), "enrichment": sm.enrichment.model_dump()}, ensure_ascii=False)
