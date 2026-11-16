"""Camada LLM com backends plugáveis, todos com o mesmo contrato: Structured Outputs.

`LLMClient.structured(system, user, schema=<PydanticModel>)` devolve SEMPRE uma
instância validada do schema, ou levanta `LLMError` (quem chama cai para o
caminho determinístico).

Backends (`LLM_BACKEND`):
- `claude_code` (padrão): usa o **Claude Code instalado localmente** em modo
  headless (`claude -p --output-format json --json-schema ...`). A autenticação é
  a do próprio Claude Code (login com sua conta Claude), sem API key no projeto.
- `anthropic_api`: SDK oficial (`messages.parse(output_format=...)`), requer
  `ANTHROPIC_API_KEY`.
- `none`: desliga LLM (planner heurístico + enriquecimento por template).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .observability import get_logger, span

T = TypeVar("T", bound=BaseModel)
log = get_logger("llm")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_API_MODEL = "claude-opus-5-5"
AUTH_HINT = "faça login no Claude Code uma vez: rode `claude` no terminal e use /login"


class LLMError(RuntimeError):
    pass


class LLMAuthError(LLMError):
    """Falha de autenticação: não adianta tentar de novo nesta execução."""


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...

    def structured(self, *, model: str, system: str, user: str, schema: type[T], effort: str,
                   max_tokens: int) -> tuple[T, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Claude Code local (headless)
# ---------------------------------------------------------------------------
def find_claude_binary(explicit: str | None = None) -> str | None:
    """Procura o executável do Claude Code: CLAUDE_CODE_BIN > PATH > instalação do app desktop (Windows)."""
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (Path.home() / ".local" / "bin" / "claude", Path.home() / ".claude" / "local" / "claude"):
        if candidate.exists():
            return str(candidate)
    local = os.environ.get("LOCALAPPDATA")
    if local:  # Claude desktop no Windows embute o Claude Code numa pasta versionada
        hits = glob.glob(os.path.join(local, "Claude-Data", "*", "claude-code", "*", "*", "claude.exe"))
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


Runner = Callable[[list[str], str, float], subprocess.CompletedProcess]


def _default_runner(cmd: list[str], stdin: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        # diretório neutro: evita que o Claude Code carregue CLAUDE.md/config deste projeto
        cwd=tempfile.gettempdir(),
    )


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(self, settings: Settings, runner: Runner | None = None):
        self.settings = settings
        self.runner = runner or _default_runner
        self._bin = find_claude_binary(settings.claude_code_bin)

    def available(self) -> bool:
        return self._bin is not None or self.runner is not _default_runner

    def structured(self, *, model, system, user, schema, effort, max_tokens):
        if not self.available():
            raise LLMError("executável `claude` não encontrado (instale o Claude Code ou defina CLAUDE_CODE_BIN)")
        cmd = [
            self._bin or "claude", "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(schema.model_json_schema(), ensure_ascii=False),
            "--system-prompt", system,
            "--tools", "",                 # sem ferramentas: só raciocínio + resposta estruturada
            "--strict-mcp-config",         # não carrega servidores MCP do usuário
            "--no-session-persistence",    # não polui o histórico de sessões
            "--effort", effort,
        ]
        if model:
            cmd += ["--model", model]
        try:
            proc = self.runner(cmd, user, self.settings.llm_timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"Claude Code excedeu {self.settings.llm_timeout_s:.0f}s") from exc
        except OSError as exc:
            raise LLMError(f"falha ao executar o Claude Code: {exc}") from exc

        try:
            out = json.loads(proc.stdout)
        except (ValueError, TypeError) as exc:
            msg = (proc.stderr or proc.stdout or "").strip()[:300]
            raise LLMError(f"saída inválida do Claude Code (exit {proc.returncode}): {msg}") from exc

        meta = {
            "served_by": next(iter(out.get("modelUsage") or {}), model or "claude-code-default"),
            "stop_reason": out.get("stop_reason"),
            "input_tokens": (out.get("usage") or {}).get("input_tokens"),
            "output_tokens": (out.get("usage") or {}).get("output_tokens"),
            "duration_api_ms": out.get("duration_api_ms"),
        }
        if out.get("is_error"):
            result = str(out.get("result", ""))
            if "login" in result.lower() or "auth" in result.lower() or out.get("api_error_status") in (401, 403):
                raise LLMAuthError(f"Claude Code não autenticado ({result}); {AUTH_HINT}")
            raise LLMError(f"Claude Code retornou erro: {result[:300]}")

        payload = out.get("structured_output")
        if payload is None:  # fallback: resposta em texto contendo o JSON
            text = str(out.get("result", "")).strip().removeprefix("```json").removesuffix("```").strip()
            try:
                payload = json.loads(text)
            except ValueError as exc:
                raise LLMError("Claude Code não devolveu JSON estruturado") from exc
        try:
            return schema.model_validate(payload), meta
        except ValidationError as exc:
            raise LLMError(f"resposta fora do schema {schema.__name__}: {exc.error_count()} erros") from exc


# ---------------------------------------------------------------------------
# API Anthropic (SDK oficial)
# ---------------------------------------------------------------------------
class AnthropicAPIBackend:
    name = "anthropic_api"

    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self._client = client

    def available(self) -> bool:
        return self._client is not None or bool(self.settings.anthropic_api_key)

    def structured(self, *, model, system, user, schema, effort, max_tokens):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key, max_retries=3)
        model = model or DEFAULT_API_MODEL
        try:
            resp = self._client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                # system estável primeiro -> elegível a prompt caching entre chamadas
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMAuthError(f"API key inválida: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limit: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"conexão: {exc}") from exc

        usage = getattr(resp, "usage", None)
        meta = {
            "served_by": getattr(resp, "model", model),
            "stop_reason": resp.stop_reason,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
        }
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMError(f"recusa do modelo ({getattr(details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise LLMError("resposta truncada (max_tokens)")
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise LLMError("resposta sem parsed_output")
        return parsed, meta


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------
class LLMClient:
    """Escolhe o backend pela configuração e aplica um circuit breaker de autenticação:
    depois de um erro de login, as próximas chamadas falham imediatamente (sem gastar
    um subprocesso/requisição por beatmap)."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None,
                 runner: Runner | None = None, backend: Backend | None = None):
        self.settings = settings or get_settings()
        if backend is not None:
            self.backend: Backend | None = backend
        elif client is not None:
            self.backend = AnthropicAPIBackend(self.settings, client)
        elif runner is not None:
            self.backend = ClaudeCodeBackend(self.settings, runner)
        elif not self.settings.llm_enabled or self.settings.llm_backend == "none":
            self.backend = None
        elif self.settings.llm_backend == "anthropic_api":
            self.backend = AnthropicAPIBackend(self.settings)
        else:
            self.backend = ClaudeCodeBackend(self.settings)
        self._auth_failed: str | None = None
        self._lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "none"

    def label(self, model: str) -> str:
        """Rótulo de proveniência gravado em resultados/cache, ex.: 'llm:claude_code:default'."""
        return f"llm:{self.backend_name}:{model or 'default'}"

    @property
    def available(self) -> bool:
        return self.backend is not None and self.backend.available() and self._auth_failed is None

    def structured(self, *, model: str, system: str, user: str, schema: type[T], purpose: str,
                   effort: str | None = None, max_tokens: int | None = None) -> T:
        if self._auth_failed:
            raise LLMAuthError(self._auth_failed)
        if not self.available:
            raise LLMError(f"LLM indisponível (backend: {self.backend_name})")
        assert self.backend is not None
        with span("llm.structured", backend=self.backend.name, purpose=purpose, model=model or "default",
                  schema=schema.__name__) as sp:
            try:
                parsed, meta = self.backend.structured(
                    model=model, system=system, user=user, schema=schema,
                    effort=effort or self.settings.llm_effort,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                )
            except LLMAuthError as exc:
                with self._lock:
                    self._auth_failed = str(exc)
                log.error("%s", exc)
                raise
            sp.set(**meta)
            return parsed
