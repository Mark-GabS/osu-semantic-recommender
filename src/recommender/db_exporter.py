"""Leitura/escrita binária do `collection.db` do osu!stable.

Formato (little-endian):
    Int32   versão (data no formato yyyymmdd, ex.: 20150203)
    Int32   número de coleções
    para cada coleção:
        String  nome
        Int32   número de beatmaps
        String  MD5 de cada beatmap (hex minúsculo, 32 chars)

`String` do osu!: byte 0x00 (string vazia/nula) ou 0x0b + ULEB128(tamanho em bytes) + UTF-8.

O osu!lazer não lê este arquivo diretamente, mas importa coleções do stable
(Settings -> Maintenance -> "Import collections from stable").
"""

from __future__ import annotations

import io
import shutil
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable

DEFAULT_VERSION = 20150203


class CollectionDBError(ValueError):
    pass


@dataclass
class Collection:
    name: str
    md5s: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ primitivas
def write_uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("ULEB128 não suporta negativos")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_uleb128(fh: BinaryIO) -> int:
    result = shift = 0
    while True:
        b = fh.read(1)
        if not b:
            raise CollectionDBError("EOF no meio de um ULEB128")
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result
        shift += 7


def write_osu_string(value: str | None) -> bytes:
    if not value:
        return b"\x00"
    data = value.encode("utf-8")
    return b"\x0b" + write_uleb128(len(data)) + data


def read_osu_string(fh: BinaryIO) -> str:
    flag = fh.read(1)
    if not flag:
        raise CollectionDBError("EOF ao ler string")
    if flag == b"\x00":
        return ""
    if flag != b"\x0b":
        raise CollectionDBError(f"marcador de string inválido: {flag!r}")
    n = read_uleb128(fh)
    data = fh.read(n)
    if len(data) != n:
        raise CollectionDBError("string truncada")
    return data.decode("utf-8")


def _read_i32(fh: BinaryIO) -> int:
    b = fh.read(4)
    if len(b) != 4:
        raise CollectionDBError("EOF ao ler Int32")
    return struct.unpack("<i", b)[0]


# ------------------------------------------------------------------ API pública
def _valid_md5(h: str) -> bool:
    return len(h) == 32 and all(c in "0123456789abcdef" for c in h)


def dumps(collections: Iterable[Collection], version: int = DEFAULT_VERSION) -> bytes:
    cols = list(collections)
    buf = io.BytesIO()
    buf.write(struct.pack("<ii", version, len(cols)))
    for col in cols:
        hashes = list(dict.fromkeys(h.lower() for h in col.md5s))  # dedup preservando ordem
        bad = [h for h in hashes if not _valid_md5(h)]
        if bad:
            raise CollectionDBError(f"MD5 inválido em '{col.name}': {bad[:3]}")
        buf.write(write_osu_string(col.name))
        buf.write(struct.pack("<i", len(hashes)))
        for h in hashes:
            buf.write(write_osu_string(h))
    return buf.getvalue()


def loads(data: bytes) -> tuple[int, list[Collection]]:
    fh = io.BytesIO(data)
    version = _read_i32(fh)
    n = _read_i32(fh)
    if n < 0:
        raise CollectionDBError("número de coleções negativo")
    cols = []
    for _ in range(n):
        name = read_osu_string(fh)
        count = _read_i32(fh)
        cols.append(Collection(name, [read_osu_string(fh) for _ in range(count)]))
    if fh.read(1):
        raise CollectionDBError("bytes extras após o fim do arquivo")
    return version, cols


def read_collection_db(path: str | Path) -> tuple[int, list[Collection]]:
    return loads(Path(path).read_bytes())


def write_collection_db(
    path: str | Path,
    collections: Iterable[Collection],
    *,
    merge_with: str | Path | None = None,
    backup: bool = True,
) -> Path:
    """Escreve `collection.db`.

    Com `merge_with`, preserva as coleções existentes do jogador: coleções de mesmo
    nome são substituídas, as demais mantidas. Se o destino já existir e `backup`
    for True, cria uma cópia `.bak-<timestamp>` antes de sobrescrever.
    """
    path = Path(path)
    new = list(collections)
    version = DEFAULT_VERSION
    final: list[Collection] = []
    if merge_with is not None and Path(merge_with).exists():
        version, existing = read_collection_db(merge_with)
        names = {c.name for c in new}
        final = [c for c in existing if c.name not in names]
    final += new
    payload = dumps(final, version)
    loads(payload)  # auto-verificação: o que escrevemos precisa ser legível
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and backup:
        shutil.copy2(path, path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}"))
    path.write_bytes(payload)
    return path
