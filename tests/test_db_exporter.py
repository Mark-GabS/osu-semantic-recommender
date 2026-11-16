import hashlib
import io
import struct

import pytest

from src.recommender.db_exporter import (
    Collection,
    CollectionDBError,
    dumps,
    loads,
    read_collection_db,
    read_uleb128,
    write_collection_db,
    write_osu_string,
    write_uleb128,
)


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


@pytest.mark.parametrize("value,encoded", [(0, b"\x00"), (1, b"\x01"), (127, b"\x7f"), (128, b"\x80\x01"),
                                           (300, b"\xac\x02"), (624485, b"\xe5\x8e\x26")])
def test_uleb128_known_vectors(value, encoded):
    assert write_uleb128(value) == encoded
    assert read_uleb128(io.BytesIO(encoded)) == value


def test_osu_string_encoding():
    assert write_osu_string("") == b"\x00"
    assert write_osu_string(None) == b"\x00"
    assert write_osu_string("abc") == b"\x0b\x03abc"
    # tamanho é em BYTES utf-8, não em caracteres
    assert write_osu_string("ジャンプ") == b"\x0b\x0c" + "ジャンプ".encode()


def test_exact_binary_layout():
    h = md5("a")
    data = dumps([Collection("X", [h])], version=20150203)
    expected = struct.pack("<ii", 20150203, 1) + b"\x0b\x01X" + struct.pack("<i", 1) + b"\x0b\x20" + h.encode()
    assert data == expected


def test_roundtrip_multiple_collections_and_unicode():
    cols = [Collection("5.2★ tech jumps 日本", [md5("a"), md5("b")]), Collection("vazia", [])]
    version, back = loads(dumps(cols))
    assert version == 20150203
    assert [(c.name, c.md5s) for c in back] == [(c.name, c.md5s) for c in cols]


def test_dedup_and_lowercase():
    h = md5("x")
    _, back = loads(dumps([Collection("c", [h.upper(), h, h])]))
    assert back[0].md5s == [h]


def test_invalid_md5_rejected():
    with pytest.raises(CollectionDBError):
        dumps([Collection("c", ["not-a-hash"])])


def test_truncated_file_rejected():
    data = dumps([Collection("c", [md5("a")])])
    with pytest.raises(CollectionDBError):
        loads(data[:-5])
    with pytest.raises(CollectionDBError):
        loads(data + b"\x00")


def test_merge_preserves_existing_and_replaces_same_name(tmp_path):
    existing = tmp_path / "collection.db"
    write_collection_db(existing, [Collection("favoritos", [md5("f")]), Collection("osr: treino", [md5("old")])],
                        backup=False)
    out = write_collection_db(existing, [Collection("osr: treino", [md5("new1"), md5("new2")])],
                              merge_with=existing)
    _, cols = read_collection_db(out)
    by_name = {c.name: c.md5s for c in cols}
    assert by_name["favoritos"] == [md5("f")]
    assert by_name["osr: treino"] == [md5("new1"), md5("new2")]
    # backup criado ao sobrescrever
    assert list(tmp_path.glob("collection.db.bak-*"))


def test_hashes_match_real_osu_files(sample_files, tmp_path):
    """O MD5 exportado tem que ser o MD5 do arquivo .osu (é assim que o osu! associa o mapa)."""
    from src.parser import parse_osu_file

    hashes = [parse_osu_file(p).md5 for p in sample_files[:5]]
    out = write_collection_db(tmp_path / "c.db", [Collection("amostra", hashes)])
    _, cols = read_collection_db(out)
    assert cols[0].md5s == [hashlib.md5(p.read_bytes()).hexdigest() for p in sample_files[:5]]
