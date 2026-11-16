"""CLI (Typer + Rich).

    python -m src.cli recommend "pedido" [--download out/osz] [--export collection.db --merge collection.db]
    python -m src.cli recommend --local "pedido"      # só nos mapas que você já tem (índice local)
    python -m src.cli index [--songs-dir PATH] [--reset] [--no-llm]
    python -m src.cli plan "pedido em linguagem natural"
    python -m src.cli inspect caminho/mapa.osu
    python -m src.cli stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import get_settings
from .observability import setup_logging

for _stream in (sys.stdout, sys.stderr):  # consoles Windows legados
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help="Recomendador semântico e mecânico de beatmaps de osu!")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="logs DEBUG (inclui spans)")) -> None:
    s = get_settings()
    setup_logging("DEBUG" if verbose else s.log_level, s.traces_file)


def _components(use_llm: bool):
    from .indexer import VectorStore, make_embedder
    from .recommender import HybridSearcher, QueryPlanner

    s = get_settings()
    emb = make_embedder(s)
    store = VectorStore(s, embedder_name=emb.name)
    return s, store, QueryPlanner(s, use_llm=use_llm), HybridSearcher(store, emb, s)


@app.command()
def index(
    songs_dir: Optional[Path] = typer.Option(None, help="pasta Songs do osu! (padrão: SONGS_DIR)"),
    reset: bool = typer.Option(False, help="apaga o índice antes de indexar"),
    no_llm: bool = typer.Option(False, "--no-llm", help="enriquecimento por template, sem chamadas LLM"),
) -> None:
    """Parseia .osu, extrai features, enriquece e indexa no banco vetorial."""
    from .indexer import index_songs

    from .llm import LLMClient

    s = get_settings()
    llm = LLMClient(s)
    use_llm = not no_llm and llm.available
    console.print(f"[bold]Indexando[/] {songs_dir or s.songs_dir}  "
                  f"(enriquecimento: {llm.label(s.enrichment_model) if use_llm else 'template'}, "
                  f"embeddings: {s.embedding_backend})")
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("processando", total=1)
        report = index_songs(songs_dir, settings=s, use_llm=use_llm, reset=reset,
                             progress=lambda i, n: prog.update(task, completed=i, total=n))
        prog.update(task, completed=1, total=1)
    t = Table(show_header=False)
    for k, v in [("arquivos lidos", report.scanned), ("indexados agora", report.indexed),
                 ("já no índice", report.skipped_existing), ("modo != standard", report.skipped_mode),
                 ("duplicados (mesmo MD5)", report.skipped_duplicates),
                 ("erros", len(report.errors)), ("enriquecimento", report.enrichment_stats)]:
        t.add_row(k, str(v))
    console.print(t)
    for err in report.errors[:10]:
        console.print(f"[red]erro[/] {err}")


@app.command()
def plan(prompt: str, no_llm: bool = typer.Option(False, "--no-llm")) -> None:
    """Mostra a HybridQuery gerada para um pedido (debug do planner)."""
    from .recommender import QueryPlanner

    q = QueryPlanner(get_settings(), use_llm=not no_llm).plan(prompt)
    console.print_json(q.model_dump_json(exclude_none=True))


@app.command()
def recommend(
    prompt: str,
    no_llm: bool = typer.Option(False, "--no-llm", help="usa o planner heurístico (sem Claude)"),
    limit: Optional[int] = typer.Option(None, help="sobrescreve o nº de mapas"),
    local: bool = typer.Option(False, "--local", help="busca no índice local (pasta Songs) em vez do site"),
    download: Optional[Path] = typer.Option(None, help="baixa os .osz recomendados nesta pasta"),
    include_owned: bool = typer.Option(False, "--include-owned", help="não esconder mapsets que você já tem"),
    enrich: bool = typer.Option(False, "--enrich", help="descreve os candidatos com o Claude (mais lento)"),
    export: Optional[Path] = typer.Option(None, help="grava um collection.db neste caminho"),
    merge: Optional[Path] = typer.Option(None, help="collection.db existente a preservar no export"),
    name: Optional[str] = typer.Option(None, help="nome da coleção (padrão: sugerido pelo planner)"),
    as_json: bool = typer.Option(False, "--json", help="saída JSON (para scripts)"),
) -> None:
    """Pedido em linguagem natural -> mapas do osu.ppy.sh ranqueados e explicados -> links/.osz/collection.db."""
    from .observability import span
    from .recommender import Collection, QueryPlanner, write_collection_db

    s = get_settings()
    planner = QueryPlanner(s, use_llm=not no_llm)
    with span("recommend", prompt=prompt, mode="local" if local else "online") as sp:
        q = planner.plan(prompt)
        if limit:
            q.limit = limit
        if local:
            _, store, _, searcher = _components(use_llm=False)
            if store.count() == 0:
                console.print("[red]Índice local vazio.[/] Rode primeiro: python -m src.cli index")
                raise typer.Exit(1)
            res = searcher.search(q)
            rows = [dict(rec=r, url=None, set=None) for r in res.recommendations]
            relaxations, where, stats = res.relaxations, res.where, {}
        else:
            from .catalog import CatalogError
            from .indexer import Enricher
            from .recommender.online_search import OnlineRecommender, osu_filter_string

            try:
                rec = OnlineRecommender(s, exclude_owned=not include_owned,
                                        enricher=Enricher(s) if enrich else None)
                with console.status("buscando no catálogo do osu! e analisando os mapas..."):
                    res = rec.search(q)
            except CatalogError as exc:
                console.print(f"[red]Catálogo indisponível:[/] {exc}")
                raise typer.Exit(1)
            rows = [dict(rec=r.rec, url=r.url, set=r.set) for r in res.recommendations]
            relaxations, where, stats = res.relaxations, osu_filter_string(q) or "(nenhum)", res.stats
        sp.set(planner=q.planner, returned=len(rows), relaxations=relaxations)

    if as_json:
        print(json.dumps({
            "query": q.model_dump(), "relaxations": relaxations, "stats": stats,
            "results": [{"md5": r["rec"].md5, "name": r["rec"].item.features.display_name, "url": r["url"],
                         "beatmapset_id": r["set"].id if r["set"] else None, "score": r["rec"].score,
                         "semantic": r["rec"].semantic, "mechanic": r["rec"].mechanic, "music": r["rec"].music,
                         "strict": r["rec"].strict, "reasons": r["rec"].reasons} for r in rows],
        }, ensure_ascii=False, indent=1))
    else:
        console.print(Panel.fit(
            f"[bold]planner:[/] {q.planner}\n"
            f"[bold]filtros:[/] {where if isinstance(where, str) else json.dumps(where, ensure_ascii=False)}\n"
            f"[bold]skills:[/] {q.skills}  [bold]evitar:[/] {q.avoid_skills}"
            + (f"  [bold]maior stream ≤[/] {q.max_stream_len}" if q.max_stream_len is not None else "")
            + f"\n[bold]gêneros/moods:[/] {q.genres} {q.moods}\n[bold]semântica:[/] {q.semantic_query}",
            title="HybridQuery"))
        if stats:
            console.print(f"[dim]catálogo: {stats.get('source')} · {stats.get('sets_scanned', 0)} mapsets lidos · "
                          f"{stats.get('candidates', 0)} candidatos · {stats.get('analyzed', 0)} analisados (.osu) · "
                          f"{stats.get('owned_skipped', 0)} que você já tem foram ocultados[/]")
        if relaxations:
            console.print(f"[yellow]Filtros relaxados:[/] {'; '.join(relaxations)}")
        t = Table(title=f"{len(rows)} recomendações", show_lines=not local)
        for col in ("#", "score", "beatmap", "sem", "mec", "mus", "por quê"):
            t.add_column(col, overflow="fold")
        for i, r in enumerate(rows, 1):
            rr = r["rec"]
            label = rr.item.features.display_name + (f"\n[link={r['url']}]{r['url']}[/link]" if r["url"] else "")
            t.add_row(str(i), f"{rr.score:.3f}", label, f"{rr.semantic:.2f}", f"{rr.mechanic:.2f}",
                      "-" if rr.music is None else f"{rr.music:.2f}", "; ".join(rr.reasons),
                      style=None if rr.strict else "dim")
        console.print(t)

    if download and rows and not local:
        from .catalog import download_osz

        done = 0
        sets = list({r["set"].id: r["set"] for r in rows}.values())
        for st in sets:
            try:
                with console.status(f"baixando {st.artist} - {st.title}..."):
                    out = download_osz(st, download)
                done += 1
                console.print(f"[green]✓[/] {out.name}", highlight=False)
            except Exception as exc:
                console.print(f"[red]✗[/] {st.id} {st.artist} - {st.title}: {exc}", highlight=False)
        console.print(f"{done}/{len(sets)} .osz em {download}. Abra cada .osz (duplo clique) ou mova para a "
                      "pasta Songs e aperte F5 na seleção de músicas do osu!.", highlight=False)

    if export and rows:
        col = Collection(name or q.collection_name, [r["rec"].md5 for r in rows])
        out = write_collection_db(export, [col], merge_with=merge)
        console.print(f"[green]collection.db gravado:[/] {out}  (coleção '{col.name}', {len(col.md5s)} mapas)",
                      highlight=False)
        if not local:
            console.print("[dim]A coleção aponta para os mapas pelo MD5; ela aparece completa no jogo depois "
                          "que os .osz forem importados.[/]")


@app.command()
def inspect(path: Path) -> None:
    """Mostra as features determinísticas de um .osu."""
    from .indexer import template_enrichment
    from .parser import extract_features, parse_osu_file

    f = extract_features(parse_osu_file(path))
    console.print_json(f.model_dump_json())
    console.print(Panel(template_enrichment(f).summary, title="descrição (template)"))


@app.command()
def doctor() -> None:
    """Verifica o backend LLM (Claude Code local ou API) com uma chamada estruturada mínima."""
    from pydantic import BaseModel

    from .llm import ClaudeCodeBackend, LLMClient, LLMError

    class Ping(BaseModel):
        ok: bool
        star_rating: float

    s = get_settings()
    llm = LLMClient(s)
    console.print(f"backend: [bold]{llm.backend_name}[/]")
    if isinstance(llm.backend, ClaudeCodeBackend):
        console.print(f"executável: {llm.backend._bin or '[red]não encontrado[/] (instale o Claude Code ou defina CLAUDE_CODE_BIN)'}")
    if not llm.available:
        console.print("[yellow]LLM indisponível: o sistema roda no modo determinístico (heurística + template).[/]")
        raise typer.Exit(1)
    try:
        r = llm.structured(model=s.planner_model, system="Answer with the requested JSON only.",
                           user="Set ok=true and star_rating to the number in: 'mapas de 5.2 estrelas'",
                           schema=Ping, purpose="doctor", effort="low")
    except LLMError as exc:
        console.print(f"[red]falhou:[/] {exc}")
        raise typer.Exit(1)
    console.print(f"[green]ok[/] — resposta estruturada: {r.model_dump()}")


@app.command()
def stats() -> None:
    """Resumo do índice."""
    s, store, *_ = _components(use_llm=False)
    items = store.all()
    console.print(f"[bold]{len(items)}[/] beatmaps indexados em {s.chroma_dir} (embedder: {store.embedder_name})")
    if not items:
        return
    srs = sorted(i.features.star_rating for i in items)
    sources: dict[str, int] = {}
    skills: dict[str, int] = {}
    for i in items:
        sources[i.enrichment.source] = sources.get(i.enrichment.source, 0) + 1
        for sk in i.features.skills:
            skills[sk] = skills.get(sk, 0) + 1
    console.print(f"SR: min {srs[0]:.2f} · mediana {srs[len(srs) // 2]:.2f} · max {srs[-1]:.2f}")
    console.print(f"enriquecimento: {sources}")
    console.print(f"skills: {dict(sorted(skills.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    app()
