"""Fontes do catálogo online de beatmaps.

- `CatboySource` (padrão, sem conta): mirror público com o formato da API v2. Entende
  os filtros nativos do osu! na query (`stars>5 bpm>170`), mas em nível de MAPSET
  (basta uma dificuldade passar) e ignora gênero/idioma; por isso o recomendador
  sempre refiltra por dificuldade do lado do cliente.
- `OfficialSource`: API v2 oficial (`/api/v2/beatmapsets/search`), com app OAuth
  gratuito (client credentials). Filtros documentados, inclusive gênero (`g`) e
  idioma (`l`). Ativada quando OSU_CLIENT_ID/OSU_CLIENT_SECRET estão configurados.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from ..config import Settings, get_settings
from ..observability import get_logger, span
from .models import STATUS_IDS, OnlineSet

log = get_logger("catalog")
USER_AGENT = "osu-semantic-recommender/0.1"


class CatalogError(RuntimeError):
    pass


class CatalogSource(Protocol):
    name: str

    def search(self, query: str, *, status: str, max_sets: int, genre_id: int | None = None,
               language_id: int | None = None) -> list[OnlineSet]: ...


def _get_json(client: httpx.Client, url: str, params: dict, retries: int = 2) -> object:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                raise CatalogError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, CatalogError, ValueError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise CatalogError(f"falha consultando {url}: {last}")


class CatboySource:
    name = "catboy"
    supports_genre_filter = False
    URL = "https://catboy.best/api/v2/search"
    PAGE = 100

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=40, follow_redirects=True)

    def search(self, query, *, status, max_sets, genre_id=None, language_id=None):
        sets: list[OnlineSet] = []
        offset = 0
        while len(sets) < max_sets:
            params: dict = {"q": query, "limit": self.PAGE, "offset": offset, "mode": 0}
            if status != "any":
                params["status"] = STATUS_IDS[status]
            with span("catalog.page", source=self.name, offset=offset):
                page = _get_json(self.client, self.URL, params)
            if not isinstance(page, list) or not page:
                break
            sets += [OnlineSet.from_api(s) for s in page]
            if len(page) < self.PAGE:
                break
            offset += self.PAGE
        return sets[:max_sets]


class OfficialSource:
    name = "osu-api"
    supports_genre_filter = True
    TOKEN_URL = "https://osu.ppy.sh/oauth/token"
    URL = "https://osu.ppy.sh/api/v2/beatmapsets/search"

    def __init__(self, client_id: str, client_secret: str, client: httpx.Client | None = None):
        self.client_id, self.client_secret = client_id, client_secret
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=40)
        self._token: str | None = None
        self._expires = 0.0

    def _auth(self) -> None:
        if self._token and time.time() < self._expires - 60:
            return
        r = self.client.post(self.TOKEN_URL, json={
            "client_id": int(self.client_id), "client_secret": self.client_secret,
            "grant_type": "client_credentials", "scope": "public",
        })
        if r.status_code != 200:
            raise CatalogError(f"autenticação na API do osu! falhou (HTTP {r.status_code}); confira OSU_CLIENT_ID/SECRET")
        data = r.json()
        self._token = data["access_token"]
        self._expires = time.time() + int(data.get("expires_in", 3600))
        self.client.headers["Authorization"] = f"Bearer {self._token}"

    def search(self, query, *, status, max_sets, genre_id=None, language_id=None):
        self._auth()
        sets: list[OnlineSet] = []
        cursor: str | None = None
        while len(sets) < max_sets:
            params: dict = {"q": query, "m": 0, "s": status}
            if genre_id:
                params["g"] = genre_id
            if language_id:
                params["l"] = language_id
            if cursor:
                params["cursor_string"] = cursor
            with span("catalog.page", source=self.name, n=len(sets)):
                page = _get_json(self.client, self.URL, params)
            items = page.get("beatmapsets", []) if isinstance(page, dict) else []
            sets += [OnlineSet.from_api(s) for s in items]
            cursor = page.get("cursor_string") if isinstance(page, dict) else None
            if not items or not cursor:
                break
        return sets[:max_sets]


def make_source(settings: Settings | None = None) -> CatalogSource:
    s = settings or get_settings()
    wants_official = s.catalog_source == "official" or (
        s.catalog_source == "auto" and s.osu_client_id and s.osu_client_secret)
    if wants_official:
        if not (s.osu_client_id and s.osu_client_secret):
            raise CatalogError("CATALOG_SOURCE=official exige OSU_CLIENT_ID e OSU_CLIENT_SECRET")
        return OfficialSource(s.osu_client_id, s.osu_client_secret)
    return CatboySource()
