"""Backend Claude Code local: testado com um runner falso (sem subprocesso real, sem rede)."""

import json
import subprocess

import pytest

from src.llm import LLMAuthError, LLMClient, LLMError
from src.recommender import HybridQuery, NumRange, QueryPlanner

PROMPT = "quero mapas de 5.2★ com jumps técnicos, rock japonês energético e poucas streams longas para treinar aim"


class FakeRunner:
    def __init__(self, payload: dict, returncode: int = 0):
        self.payload, self.returncode, self.calls = payload, returncode, []

    def __call__(self, cmd, stdin, timeout):
        self.calls.append((cmd, stdin))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=json.dumps(self.payload), stderr="")


def ok_payload(structured: dict | None = None, result: str = "") -> dict:
    out = {"type": "result", "is_error": False, "result": result, "stop_reason": "end_turn",
           "usage": {"input_tokens": 120, "output_tokens": 40}, "modelUsage": {"claude-opus-5-5": {}}}
    if structured is not None:
        out["structured_output"] = structured
    return out


EXPECTED = HybridQuery(semantic_query="energetic jrock technical jumps", collection_name="5.2* tech jumps",
                       star_rating=NumRange(min=4.9, max=5.5), skills=["jumps", "tech"])


def test_structured_output_and_command_shape(settings):
    runner = FakeRunner(ok_payload(EXPECTED.model_dump()))
    q = QueryPlanner(settings, llm=LLMClient(settings, runner=runner)).plan(PROMPT)
    assert q.star_rating == EXPECTED.star_rating and q.skills == ["jumps", "tech"]
    assert q.planner == "llm:claude_code:default"

    cmd, stdin = runner.calls[0]
    assert "-p" in cmd and cmd[cmd.index("--output-format") + 1] == "json"
    assert json.loads(cmd[cmd.index("--json-schema") + 1])["title"] == "HybridQuery"
    assert cmd[cmd.index("--tools") + 1] == ""        # sem ferramentas
    assert "--model" not in cmd                        # usa o modelo padrão do Claude Code
    assert PROMPT in stdin                             # pedido vai pelo stdin, não pela linha de comando


def test_model_override_is_passed(settings):
    s = settings.model_copy(update={"planner_model": "sonnet"})
    runner = FakeRunner(ok_payload(EXPECTED.model_dump()))
    QueryPlanner(s, llm=LLMClient(s, runner=runner)).plan(PROMPT)
    cmd, _ = runner.calls[0]
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_falls_back_to_json_in_result_text(settings):
    runner = FakeRunner(ok_payload(result="```json\n" + EXPECTED.model_dump_json() + "\n```"))
    q = QueryPlanner(settings, llm=LLMClient(settings, runner=runner)).plan(PROMPT)
    assert q.planner.startswith("llm:") and q.star_rating == EXPECTED.star_rating


def test_not_logged_in_trips_circuit_breaker_and_falls_back(settings):
    runner = FakeRunner({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"})
    llm = LLMClient(settings, runner=runner)
    planner = QueryPlanner(settings, llm=llm)
    q = planner.plan(PROMPT)
    assert q.planner == "heuristic" and q.star_rating == NumRange(min=4.9, max=5.5)
    planner.plan(PROMPT)
    assert len(runner.calls) == 1          # 2ª chamada nem dispara subprocesso
    assert not llm.available
    with pytest.raises(LLMAuthError):
        llm.structured(model="", system="s", user="u", schema=HybridQuery, purpose="t")


def test_schema_violation_is_an_error(settings):
    runner = FakeRunner(ok_payload({"star_rating": "cinco"}))
    with pytest.raises(LLMError, match="schema"):
        LLMClient(settings, runner=runner).structured(model="", system="s", user="u", schema=HybridQuery,
                                                      purpose="t")


def test_garbage_stdout_is_an_error(settings):
    def runner(cmd, stdin, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    with pytest.raises(LLMError, match="boom"):
        LLMClient(settings, runner=runner).structured(model="", system="s", user="u", schema=HybridQuery,
                                                      purpose="t")


def test_backend_none_means_deterministic(settings):
    llm = LLMClient(settings)  # fixture: llm_backend="none"
    assert not llm.available and llm.backend_name == "none"
    assert QueryPlanner(settings, llm=llm).plan(PROMPT).planner == "heuristic"
