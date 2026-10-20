"""The installed command must keep archive and scientific interfaces reachable."""
import json
from types import SimpleNamespace

import pytest

from magictools import agent_cli
from magictools.frames import cli as frames_cli


def test_catalogue_includes_every_command_family(capsys):
    assert agent_cli.main(["list"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert "fitsky" in result["tools"]
    assert "search" in result["commands"]["archive"]
    assert "frames inspect" in result["commands"]["archive"]
    assert "force_track" in result["commands"]["manifests"]
    assert "fit-sky-stream" in result["commands"]["research"]


@pytest.mark.parametrize("prefix", [[], ["frames"]])
def test_archive_capabilities_through_shared_entrypoint(prefix, capsys):
    assert agent_cli.main(prefix + ["capabilities"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["providers"]) == 49
    assert "frames inspect" in result["discoveryCommands"]


def test_inspect_namespaces_preserve_scientific_contract(tmp_path, monkeypatch, capsys):
    (tmp_path / "run.json").write_text('{"status":"completed"}')
    assert agent_cli.main(["inspect", str(tmp_path)]) == 0
    science = json.loads(capsys.readouterr().out)
    assert science["ok"] and science["result"]["status"] == "completed"
    calls = []
    monkeypatch.setattr(frames_cli, "main", lambda argv, **kwargs: calls.append((argv, kwargs)) or 3)
    assert agent_cli.main(["frames", "inspect", "alma", "--output-dir", "schema"]) == 3
    assert calls == [(["inspect", "alma", "--output-dir", "schema"], {"prog": "constellation frames"})]


def test_manifest_tool_runs_under_constellation(tmp_path, capsys):
    manifest = tmp_path / "nulls.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "hypotheses": [{"id": "a", "prior_weight": 1}, {"id": "b", "prior_weight": 1}],
        "non_detections": [{"id": "past", "detection_probability": [.9, .1]}],
    }))
    assert agent_cli.main(["learn_null", str(manifest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["posterior_weights"] == pytest.approx([.1, .9])


@pytest.mark.parametrize("command,module", agent_cli.RESEARCH_COMMANDS.items())
def test_research_dispatch_preserves_arguments_and_exit_code(command, module, monkeypatch):
    calls = []
    def imported(name):
        assert name == module
        return SimpleNamespace(main=lambda argv, **kwargs: calls.append((argv, kwargs)) or 2)
    monkeypatch.setattr(agent_cli, "import_module", imported)
    assert agent_cli.main([command, "input.json", "output", "--help"]) == 2
    assert calls == [(["input.json", "output", "--help"], {"prog": "constellation " + command})]


def test_command_families_have_no_collisions():
    families = [set(agent_cli.ARCHIVE_COMMANDS), set(agent_cli.MANIFEST_COMMANDS),
                set(agent_cli.DEMO_COMMANDS), set(agent_cli.RESEARCH_COMMANDS),
                {"list", "doctor", "schema", "example", "validate", "run", "inspect", "frames"}]
    assert sum(map(len, families)) == len(set().union(*families))


def test_namespaced_help_uses_public_command(capsys):
    with pytest.raises(SystemExit) as exited:
        agent_cli.main(["frames", "inspect", "--help"])
    assert exited.value.code == 0
    assert "constellation frames inspect" in capsys.readouterr().out
