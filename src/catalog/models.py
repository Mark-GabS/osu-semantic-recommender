"""Modelos do catálogo online (formato da API v2 do osu!, também usado pelos mirrors)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# IDs oficiais de gênero/idioma do osu! (https://osu.ppy.sh/docs -> Beatmapset search)
GENRES = {1: "unspecified", 2: "video game", 3: "anime", 4: "rock", 5: "pop", 6: "other", 7: "novelty",
          9: "hip hop", 10: "electronic", 11: "metal", 12: "classical", 13: "folk", 14: "jazz"}
LANGUAGES = {1: "unspecified", 2: "english", 3: "japanese", 4: "chinese", 5: "instrumental", 6: "korean",
             7: "french", 8: "german", 9: "swedish", 10: "spanish", 11: "italian", 12: "russian",
             13: "polish", 14: "other"}
STATUS_IDS = {"graveyard": -2, "wip": -1, "pending": 0, "ranked": 1, "approved": 2, "qualified": 3, "loved": 4}


class OnlineBeatmap(BaseModel):
    id: int
    beatmapset_id: int
    version: str
    mode: str = "osu"
    difficulty_rating: float
    bpm: float = 0.0
    ar: float = 0.0
    cs: float = 0.0
    od: float = Field(0.0, alias="accuracy")
    hp: float = Field(0.0, alias="drain")
    total_length: int = 0
    hit_length: int = 0
    checksum: str | None = None
    count_circles: int = 0
    count_sliders: int = 0
    playcount: int = 0
    status: str = ""

    model_config = {"populate_by_name": True}

    @property
    def is_standard(self) -> bool:
        return self.mode == "osu"

    def range_value(self, name: str) -> float:
        return {"star_rating": self.difficulty_rating, "bpm": self.bpm, "ar": self.ar, "cs": self.cs,
                "od": self.od, "length_s": float(self.total_length)}[name]


class OnlineSet(BaseModel):
    id: int
    artist: str
    title: str
    creator: str = ""
    source: str = ""
    tags: str = ""
    status: str = ""
    genre_id: int | None = None
    language_id: int | None = None
    favourite_count: int = 0
    play_count: int = 0
    ranked_date: str | None = None
    preview_url: str | None = None
    beatmaps: list[OnlineBeatmap] = Field(default_factory=list)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "OnlineSet":
        data = dict(raw)
        # a API às vezes manda o objeto (genre: {id, name}) e às vezes só o id
        for key in ("genre", "language"):
            obj = data.get(key)
            if isinstance(obj, dict) and data.get(f"{key}_id") is None:
                data[f"{key}_id"] = obj.get("id")
        data["beatmaps"] = [b for b in (data.get("beatmaps") or []) if isinstance(b, dict)]
        return cls.model_validate(data)

    @property
    def url(self) -> str:
        return f"https://osu.ppy.sh/beatmapsets/{self.id}"

    @property
    def genre(self) -> str:
        return GENRES.get(self.genre_id or 1, "unspecified")

    @property
    def language(self) -> str:
        return LANGUAGES.get(self.language_id or 1, "unspecified")

    def beatmap_url(self, b: OnlineBeatmap) -> str:
        return f"{self.url}#osu/{b.id}"
