import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import sb.cfg
import sb.io
import sb.parsing
import sb.runners
import sb.smartbugs
import sb.tools
from sb.settings import Settings


def test_llm_tool_loads_through_normal_tool_mechanism():
    tools = sb.tools.load(["llm"])
    assert len(tools) == 1
    assert tools[0].id == "llm"
    assert tools[0].mode == "solidity"
    assert tools[0].runner == "llm"
    assert tools[0].image is None


def test_native_llm_task_does_not_load_docker_image(monkeypatch, tmp_path):
    contract = tmp_path / "A.sol"
    contract.write_text("contract A {}", encoding="utf-8")
    tool = sb.tools.load(["llm"])[0]
    settings = SimpleNamespace(
        runtime=False,
        main=False,
        continue_on_errors=False,
        resultdir=lambda *args: str(tmp_path / "result"),
    )
    monkeypatch.setattr(
        sb.smartbugs.sb.docker,
        "is_loaded",
        lambda image: (_ for _ in ()).throw(AssertionError("Docker must not be used")),
    )
    tasks = sb.smartbugs.collect_tasks([(str(contract), "A.sol")], [tool], settings)
    assert len(tasks) == 1
    assert tasks[0].tool.runner == "llm"


def test_runner_dispatches_to_llm(monkeypatch):
    task = SimpleNamespace(tool=SimpleNamespace(runner="llm"))
    expected = (0, [], b"{}", [], {})
    import sb.llm.runner

    monkeypatch.setattr(sb.llm.runner, "execute", lambda value: expected)
    assert sb.runners.execute(task) == expected


def test_tool_parser_produces_normalized_finding():
    parser_path = Path(sb.cfg.TOOLS_HOME) / "llm" / "parser.py"
    spec = importlib.util.spec_from_file_location("test_llm_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = json.dumps(
        {
            "status": "completed",
            "findings": [{"name": "reentrancy", "line": 4}],
            "parse_result": {"warnings": []},
            "technical_error": None,
        }
    ).encode()
    findings, infos, errors, fails = module.parse(0, [], output)
    assert findings == [{"name": "reentrancy", "line": 4}]
    assert infos == errors == fails == set()


def test_normal_smartbugs_parser_accepts_llm_output():
    sb.parsing.tool_parsers.clear()
    task_log = {
        "filename": "A.sol",
        "tool": {"id": "llm", "mode": "solidity", "parser": "parser.py"},
        "result": {"exit_code": 0},
    }
    output = json.dumps(
        {
            "status": "completed",
            "findings": [{"name": "reentrancy", "line": 4}],
            "parse_result": {"warnings": []},
            "technical_error": None,
        }
    ).encode()
    parsed = sb.parsing.parse(task_log, [], output)
    assert parsed["findings"][0]["filename"] == "A.sol"
    assert parsed["findings"][0]["line"] == 4


def test_settings_accept_llm_mapping():
    settings = Settings()
    value = {"profile": "reentrancy", "strategy": "zero_shot"}
    settings.update({"llm": value})
    assert settings.llm == value


def test_llm_runner_writes_provenance_and_serialized_result(monkeypatch, tmp_path):
    import sb.llm.runner

    contract = tmp_path / "A.sol"
    contract.write_text("contract A {}", encoding="utf-8")
    task = SimpleNamespace(
        absfn=str(contract),
        relfn="A.sol",
        rdir=str(tmp_path),
        tool=SimpleNamespace(options={}),
        settings=SimpleNamespace(
            llm={
                "profile": "reentrancy",
                "strategy": "zero_shot",
                "provider": {
                    "provider": "openai",
                    "model": "model-id",
                    "api_key_env": "NOT_USED",
                },
            }
        ),
    )

    class FakeAnalyzer:
        def __init__(self, profile, provider_config):
            self.profile = profile
            self.provider_config = provider_config

        def analyze(self, **kwargs):
            parse_result = SimpleNamespace(dict=lambda: {"classification_valid": True})
            return {
                "status": "completed",
                "findings": [{"name": "reentrancy"}],
                "parse_result": parse_result,
                "provenance": {
                    "inference_id": "id-1",
                    "technical_error": None,
                },
            }

    monkeypatch.setattr(sb.llm.runner, "LLMAnalyzer", FakeAnalyzer)
    exit_code, logs, output, sb_log, details = sb.llm.runner.execute(task)
    assert exit_code == 0
    assert sb_log == []
    assert details["profile"] == "reentrancy"
    assert "status=completed" in logs[0]
    assert json.loads(output)["findings"] == [{"name": "reentrancy"}]
    provenance = sb.io.read_json(tmp_path / "llm-provenance.json")
    assert provenance["inference_id"] == "id-1"
