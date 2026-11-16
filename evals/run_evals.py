"""Suíte de avaliação do recomendador.

Mede duas coisas separadamente (para saber ONDE o sistema erra):

1. Planner (linguagem natural -> HybridQuery): acurácia campo a campo contra
   `expected` (faixas numéricas com tolerância; listas por inclusão).
2. Retrieval end-to-end (planner + busca híbrida): precision@5, recall@10,
   nDCG@5 (relevância binária definida por predicados sobre as features) e
   hit@3 das respostas-ouro (`must_top3`).

Por padrão constrói um índice efêmero e reprodutível dos mapas de exemplo
(embeddings hash + enriquecimento por template), então roda offline.

    python evals/run_evals.py                      # planner heurístico
    python evals/run_evals.py --planner llm        # planner Claude (via Claude Code local, sem API key)
    python evals/run_evals.py --planner both       # compara lado a lado
    python evals/run_evals.py --min-p5 0.6         # falha (exit 1) abaixo do limiar -> uso em CI
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.config import Settings, get_settings  # noqa: E402
from src.indexer import HashEmbedder, StoredMap, VectorStore, index_songs  # noqa: E402
from src.llm import LLMAuthError, LLMClient  # noqa: E402
from src.observability import setup_logging  # noqa: E402
from src.recommender import HeuristicQueryPlanner, HybridQuery, HybridSearcher, LLMQueryPlanner  # noqa: E402

console = Console()
DATASET = Path(__file__).resolve().parent / "test_queries.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------- planner
def _range_ok(pred: Any, exp: dict, tol: float) -> bool:
    for bound in ("min", "max"):
        e = exp.get(bound)
        p = getattr(pred, bound, None) if pred is not None else None
        if e is None:
            if p is not None:
                return False
        elif p is None or abs(p - e) > tol:
            return False
    return True


def score_planner(q: HybridQuery, expected: dict, tolerances: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key, exp in expected.items():
        if key in tolerances:
            out[key] = _range_ok(getattr(q, key), exp, tolerances[key])
        elif key == "max_stream_len_lte":
            out[key] = q.max_stream_len is not None and q.max_stream_len <= exp
        elif key == "skills_include":
            out[key] = set(exp) <= set(q.skills)
        elif key == "avoid_include":
            out[key] = set(exp) <= set(q.avoid_skills)
        elif key == "genres_include":
            out[key] = set(exp) <= set(q.genres)
        elif key == "moods_include":
            out[key] = set(exp) <= set(q.moods)
        elif key == "training_goal":
            out[key] = (q.training_goal or "").lower() == exp
        elif key == "limit":
            out[key] = q.limit == exp
        else:
            raise KeyError(f"campo de expected desconhecido: {key}")
    return out


# ---------------------------------------------------------------- retrieval
def is_relevant(m: StoredMap, rel: dict) -> bool:
    f, e = m.features, m.enrichment
    for key, cond in rel.items():
        if key in ("star_rating", "bpm", "ar", "cs", "length_s"):
            lo, hi = cond
            if not lo <= getattr(f, key) <= hi:
                return False
        elif key == "skills_any" and not set(cond) & set(f.skills):
            return False
        elif key == "skills_all" and not set(cond) <= set(f.skills):
            return False
        elif key == "max_stream_len_lte" and f.max_stream_len > cond:
            return False
        elif key == "genres_any" and not set(cond) & set(e.genres):
            return False
    return True


def ndcg(rels: list[int], k: int, n_relevant: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / ideal if ideal else 0.0


def build_index(settings: Settings) -> tuple[VectorStore, HashEmbedder]:
    emb = HashEmbedder(settings.hash_dim)
    store = VectorStore(settings, embedder_name=emb.name, ephemeral=True)
    index_songs(settings.songs_dir, settings=settings, use_llm=False, store=store, embedder=emb)
    return store, emb


def run(planner_name: str, settings: Settings, store: VectorStore, emb: HashEmbedder, data: dict) -> dict:
    planner = HeuristicQueryPlanner() if planner_name == "heuristic" else LLMQueryPlanner(settings)
    searcher = HybridSearcher(store, emb, settings)
    universe = store.all()
    rows = []
    for item in data["queries"]:
        t0 = time.perf_counter()
        error = None
        try:
            q = planner.plan(item["prompt"])
        except LLMAuthError:
            raise  # sem login não há o que avaliar: aborta em vez de gerar 18 linhas de erro
        except Exception as exc:  # um erro de LLM vira uma linha com score 0, não aborta a suíte
            error = f"{type(exc).__name__}: {exc}"
            q = HybridQuery(semantic_query=item["prompt"], collection_name="error")
        plan_ms = (time.perf_counter() - t0) * 1000
        fields = score_planner(q, item["expected"], data["tolerances"])
        res = searcher.search(q)
        ranked = [r.item for r in res.recommendations]
        relevant_ids = {m.features.md5 for m in universe if is_relevant(m, item["relevance"])}
        rels = [int(m.features.md5 in relevant_ids) for m in ranked]
        k5 = min(5, len(relevant_ids)) or 5
        top3 = [m.features.title for m in ranked[:3]]
        rows.append({
            "id": item["id"],
            "held_out": bool(item.get("held_out")),
            "prompt": item["prompt"],
            "planner_error": error,
            "plan_ms": round(plan_ms, 1),
            "fields": fields,
            "field_acc": sum(fields.values()) / len(fields),
            "n_relevant": len(relevant_ids),
            # precision@k com k = min(5, nº de relevantes): não pune queries com poucos relevantes
            "p@5": sum(rels[:k5]) / k5,
            "recall@10": sum(rels[:10]) / len(relevant_ids) if relevant_ids else 0.0,
            "ndcg@5": ndcg(rels, 5, len(relevant_ids)),
            "hit@3": any(t in top3 for t in item.get("must_top3", [])),
            "top3": top3,
            "relaxations": res.relaxations,
            "query": q.model_dump(exclude_none=True),
        })
    agg = {"planner": planner_name, **_aggregate(rows)}
    agg["dev"] = _aggregate([r for r in rows if not r["held_out"]])
    agg["held_out"] = _aggregate([r for r in rows if r["held_out"]])
    return {"summary": agg, "rows": rows}


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "field_accuracy": sum(r["field_acc"] for r in rows) / n,
        "exact_plans": sum(r["field_acc"] == 1.0 for r in rows) / n,
        "p@5": sum(r["p@5"] for r in rows) / n,
        "recall@10": sum(r["recall@10"] for r in rows) / n,
        "ndcg@5": sum(r["ndcg@5"] for r in rows) / n,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "avg_plan_ms": sum(r["plan_ms"] for r in rows) / n,
        "planner_errors": sum(r["planner_error"] is not None for r in rows),
    }


def print_report(result: dict) -> None:
    s = result["summary"]
    t = Table(title=f"Evals — planner: {s['planner']}")
    for col in ("id", "campos", "P@5", "R@10", "nDCG@5", "hit@3", "top-3", "campos errados"):
        t.add_column(col, overflow="fold")
    for r in result["rows"]:
        wrong = [k for k, ok in r["fields"].items() if not ok]
        t.add_row(r["id"] + (" (held-out)" if r["held_out"] else ""), f"{r['field_acc']:.0%}", f"{r['p@5']:.2f}", f"{r['recall@10']:.2f}",
                  f"{r['ndcg@5']:.2f}", "✓" if r["hit@3"] else "✗", ", ".join(r["top3"]),
                  ", ".join(wrong) + (f" [erro: {r['planner_error']}]" if r["planner_error"] else ""))
    console.print(t)
    for label, part in (("dev", s["dev"]), ("held-out", s["held_out"]), ("total", s)):
        if not part.get("n"):
            continue
        console.print(
            f"[bold]{label:>8}[/] (n={part['n']}) planner: campos {part['field_accuracy']:.1%} · "
            f"planos exatos {part['exact_plans']:.0%} · {part['avg_plan_ms']:.0f} ms · erros {part['planner_errors']}"
            f"  |  retrieval: P@5 {part['p@5']:.3f} · R@10 {part['recall@10']:.3f} · "
            f"nDCG@5 {part['ndcg@5']:.3f} · hit@3 {part['hit@3']:.0%}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["heuristic", "llm", "both"], default="heuristic")
    ap.add_argument("--min-p5", type=float, default=None, help="limiar mínimo de P@5 no split dev (exit 1 se abaixo)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    base = get_settings()
    settings = base.model_copy(update={
        "data_dir": RESULTS_DIR / ".tmp", "collection_name": f"evals_{uuid.uuid4().hex[:8]}",
        "embedding_backend": "hash", "songs_dir": ROOT / "data" / "sample_beatmaps",
    })
    setup_logging("WARNING", None)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    store, emb = build_index(settings)

    planners = ["heuristic", "llm"] if args.planner == "both" else [args.planner]
    if "llm" in planners and not LLMClient(base).available:
        console.print("[red]planner LLM indisponível.[/] Rode `python -m src.cli doctor` para diagnosticar "
                      "(backend padrão: Claude Code local).")
        return 2
    results = []
    for p in planners:
        try:
            res = run(p, settings, store, emb, data)
        except LLMAuthError as exc:
            console.print(f"[red]{exc}[/]")
            return 2
        print_report(res)
        results.append(res)
        if not args.no_save:
            RESULTS_DIR.mkdir(exist_ok=True)
            out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{p}.json"
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            console.print(f"[dim]resultado salvo em {out}[/]")

    # o gate olha o split dev: o held-out existe para medir generalização e é, de propósito,
    # difícil para o planner heurístico (que é o que o CI roda, sem LLM)
    if args.min_p5 is not None and any(r["summary"]["dev"]["p@5"] < args.min_p5 for r in results):
        console.print(f"[red]P@5 (dev) abaixo do limiar {args.min_p5}[/]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
