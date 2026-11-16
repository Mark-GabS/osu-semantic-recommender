"""Checagem ao vivo da busca ONLINE (osu.ppy.sh).

O catálogo muda todo dia, então não existe gabarito fixo de "mapas certos". Este script
mede outra coisa, que não depende do dia: os resultados CUMPREM o que o pedido exigiu?

- filtros numéricos (SR, BPM, AR...) conferidos nos valores oficiais do site;
- limite de stream conferido no .osu real (o catálogo não tem essa informação);
- gênero pedido presente no mapset (gênero/idioma do site + tags);
- skills pedidas marcadas pelo extrator mecânico.

    python evals/run_online_checks.py                 # planner heurístico
    python evals/run_online_checks.py --planner llm   # planner Claude (Claude Code local)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.observability import setup_logging  # noqa: E402
from src.recommender import HeuristicQueryPlanner, LLMQueryPlanner  # noqa: E402
from src.recommender.hybrid_search import RANGE_FIELDS, expand_genres  # noqa: E402
from src.recommender.online_search import OnlineRecommender  # noqa: E402

console = Console()
PROMPTS = [
    "quero mapas de 5.2★ com jumps técnicos, rock japonês energético e poucas streams longas para treinar aim",
    "deathstreams de metal acima de 6 estrelas",
    "mapas de anime entre 4 e 5 estrelas com AR 9, sem streams",
    "eletrônica de 180 bpm com streams para treinar velocidade, até 5.5 estrelas",
    "j-pop alegre com jumps, 4.5 estrelas",
    "mapas técnicos de música clássica entre 4 e 6 estrelas",
]


def check(res) -> dict:
    q, recs = res.query, res.recommendations
    n = len(recs) or 1
    ranges = [name for name in RANGE_FIELDS if getattr(q, name) is not None]
    numeric_ok = sum(all(getattr(q, k).contains(r.beatmap.range_value(k)) for k in ranges) for r in recs)
    stream_ok = sum(q.max_stream_len is None or r.rec.item.features.max_stream_len <= q.max_stream_len for r in recs)
    genre_ok = sum(bool(set(q.genres) & expand_genres(r.rec.item.enrichment.genres)) for r in recs) if q.genres else None
    skill_ok = sum(any(s in r.rec.item.features.skills for s in q.skills) for r in recs) if q.skills else None
    return {
        "returned": len(recs),
        "numeric": numeric_ok / n,
        "stream_limit": stream_ok / n,
        "genre": None if genre_ok is None else genre_ok / n,
        "skills_any": None if skill_ok is None else skill_ok / n,
        "relaxed": sum(not r.rec.strict for r in recs) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    setup_logging("WARNING", None)
    s = get_settings()
    planner = HeuristicQueryPlanner() if args.planner == "heuristic" else LLMQueryPlanner(s)
    rec = OnlineRecommender(s)
    rows = []
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        q = planner.plan(prompt)
        q.limit = args.limit
        res = rec.search(q)
        rows.append({"prompt": prompt, "seconds": round(time.perf_counter() - t0, 1), **check(res),
                     "top3": [f"{r.set.artist} - {r.set.title} [{r.beatmap.version}]" for r in res.recommendations[:3]]})

    fmt = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
    t = Table(title=f"Checagem online — planner {args.planner} (top {args.limit})")
    for col in ("pedido", "n", "faixas", "stream", "gênero", "skills", "relaxados", "s"):
        t.add_column(col, overflow="fold")
    for r in rows:
        t.add_row(r["prompt"][:60], str(r["returned"]), fmt(r["numeric"]), fmt(r["stream_limit"]), fmt(r["genre"]),
                  fmt(r["skills_any"]), fmt(r["relaxed"]), f"{r['seconds']:.0f}")
    console.print(t)
    out = ROOT / "evals" / "results" / f"online-{args.planner}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"[dim]salvo em {out}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
