"""Observabilidade leve: logging com Rich + spans de trace gravados em JSONL.

Cada operação relevante (parse, enrichment, chamada LLM, busca) abre um `span`.
O span mede latência, carrega atributos (modelo, tokens, nº de candidatos...) e é
persistido em `<DATA_DIR>/traces.jsonl`, o que permite análise posterior com
`jq`/pandas sem depender de um backend externo.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.logging import RichHandler

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_parent_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("parent_span", default=None)
_configured = False
_traces_file: Path | None = None


def setup_logging(level: str = "INFO", traces_file: Path | None = None) -> None:
    """Configura logging uma única vez por processo."""
    global _configured, _traces_file
    _traces_file = traces_file
    if _configured:
        logging.getLogger().setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        # logs em stderr: mantém stdout limpo para `--json`
        handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True, show_path=False, markup=False)],
    )
    for noisy in ("httpx", "chromadb", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"osr.{name}")


def _write(record: dict[str, Any]) -> None:
    if _traces_file is None:
        return
    try:
        _traces_file.parent.mkdir(parents=True, exist_ok=True)
        with _traces_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:  # observabilidade nunca deve derrubar o pipeline
        pass


class Span:
    def __init__(self, name: str, attrs: dict[str, Any]):
        self.name = name
        self.attrs = attrs
        self.span_id = uuid.uuid4().hex[:12]

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Abre um span. Spans aninhados herdam trace_id e registram o parent."""
    trace_token = None
    if _trace_id.get() is None:
        trace_token = _trace_id.set(uuid.uuid4().hex[:16])
    s = Span(name, dict(attrs))
    parent = _parent_span.get()
    parent_token = _parent_span.set(s.span_id)
    start = time.perf_counter()
    status = "ok"
    try:
        yield s
    except Exception as exc:
        status = "error"
        s.set(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": _trace_id.get(),
            "span_id": s.span_id,
            "parent_id": parent,
            "name": name,
            "status": status,
            "duration_ms": duration_ms,
            **s.attrs,
        }
        _write(record)
        get_logger("trace").debug("%s %.1fms %s", name, duration_ms, s.attrs)
        _parent_span.reset(parent_token)
        if trace_token is not None:
            _trace_id.reset(trace_token)
