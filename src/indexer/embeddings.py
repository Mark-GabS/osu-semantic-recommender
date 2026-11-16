"""Embeddings plugáveis.

- `HashEmbedder`: determinístico, offline, sem download. Hashing de unigramas,
  bigramas e trigramas de caracteres + normalização bilíngue PT/EN de termos do
  domínio (ex.: "japonês" -> "japanese", "energético" -> "energetic"). É o padrão
  para testes e evals reprodutíveis.
- `FastEmbedder`: modelo multilíngue ONNX via `fastembed` (opcional), melhor
  generalização semântica para linguagem natural livre.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Protocol

import numpy as np

# Normalização de vocabulário do domínio (PT/EN/gírias de osu!) para um termo canônico.
CANON: dict[str, str] = {
    # gêneros
    "japones": "japanese", "japonesa": "japanese", "jp": "japanese", "nihon": "japanese",
    "jrock": "jrock", "j-rock": "jrock", "rock japones": "jrock",
    "jpop": "jpop", "j-pop": "jpop", "jmetal": "metal", "j-metal": "metal",
    "eletronica": "electronic", "eletronico": "electronic", "edm": "electronic",
    "classica": "classical", "classico": "classical", "orquestral": "orchestral",
    "piano": "piano", "metal": "metal", "pesado": "heavy", "anime": "anime",
    "vocaloid": "vocaloid", "lofi": "lofi", "lo-fi": "lofi",
    # moods
    "energetico": "energetic", "energetica": "energetic", "animado": "energetic", "animada": "energetic",
    "agitado": "energetic", "agitada": "energetic", "intenso": "intense", "intensa": "intense",
    "energeticos": "energetic", "energeticas": "energetic", "animados": "energetic", "animadas": "energetic",
    "agitados": "energetic", "agitadas": "energetic", "intensos": "intense", "intensas": "intense",
    "calmo": "calm", "calma": "calm", "calmos": "calm", "calmas": "calm", "tranquilo": "calm",
    "tranquila": "calm", "tranquilos": "calm", "tranquilas": "calm", "relaxante": "relaxing",
    "relaxantes": "relaxing", "chill": "calm",
    "triste": "sad", "tristes": "sad", "emocional": "emotional", "emocionais": "emotional",
    "emocionante": "emotional", "feliz": "happy", "felizes": "happy", "alegre": "happy", "alegres": "happy",
    "agressivo": "aggressive", "agressiva": "aggressive", "agressivos": "aggressive", "agressivas": "aggressive",
    "fofo": "cute", "fofa": "cute", "fofos": "cute", "fofas": "cute",
    # mecânicas
    "pulos": "jumps", "pulo": "jumps", "jump": "jumps", "saltos": "jumps", "mira": "aim",
    "tecnico": "technical", "tecnicos": "technical", "tecnica": "technical", "tecnicas": "technical",
    "tech": "technical", "stream": "streams", "streams": "streams", "rajadas": "bursts", "burst": "bursts",
    "resistencia": "stamina", "velocidade": "speed", "rapido": "fast", "rapida": "fast",
    "leitura": "reading", "fluido": "flow", "fluidos": "flow", "flow": "flow",
    "deathstream": "deathstream", "deathstreams": "deathstream", "treinar": "training", "treino": "training",
    "jumps": "jumps", "aim": "aim", "classicos": "classical", "classicas": "classical",
    "eletronicos": "electronic", "eletronicas": "electronic",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_tokens(text: str) -> list[str]:
    text = _strip_accents(text.lower())
    text = text.replace("rock japones", "jrock").replace("pop japones", "jpop")
    out = []
    for tok in _TOKEN_RE.findall(text):
        tok = CANON.get(tok, tok)
        out.append(tok.replace("-", ""))
    return out


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = normalize_tokens(text)
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.6) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.15) for i in range(len(padded) - 2)]
        for feat, w in feats:
            idx, sign = self._bucket(feat)
            vec[idx] += sign * w
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # import tardio: dependência opcional

        self._model = TextEmbedding(model_name=model_name)
        self.name = f"fastembed:{model_name}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for v in self._model.embed(texts):
            arr = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(arr)) or 1.0
            out.append((arr / n).tolist())
        return out


def get_embedder(backend: str, *, dim: int = 512, model_name: str = "") -> Embedder:
    if backend == "fastembed":
        return FastEmbedder(model_name)
    return HashEmbedder(dim)


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    den = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(va @ vb / den)


def is_finite(vec: list[float]) -> bool:
    return all(math.isfinite(x) for x in vec)
