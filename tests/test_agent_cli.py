"""Agent protocol and real engine smoke tests, including interruption recovery."""

import json
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator

from magictools import agent_cli
from magictools.agent_contract import TOOLS, schema


def call(capsys, *args):
    code = agent_cli.main(list(map(str, args)))
    captured = capsys.readouterr()
    value = json.loads(captured.out)  # Fails on progress/extra JSON on stdout.
    assert value["schema_version"] == 1
    return code, value, captured.err


def test_discovery_schemas_and_usage_errors(capsys):
    code, result, _ = call(capsys, "list")
    assert code == 0 and set(result["result"]["tools"]) == set(TOOLS)
    for tool in TOOLS:
        Draft202012Validator.check_schema(schema(tool))
        code, result, _ = call(capsys, "schema", tool)
        assert code == 0 and result["result"]["title"] == tool
    code, result, _ = call(capsys, "does-not-exist")
    assert code == 2 and result["error"]["code"] == "invalid_input"
    code, result, _ = call(capsys, "doctor")
    assert code == 0 and "flint" in result["result"]["tools"]["fit_sky"]


@pytest.mark.parametrize("tool", list(TOOLS))
def test_generated_examples_validate_and_execute(tool, tmp_path, capsys, monkeypatch):
    pytest.importorskip("flint")
    pytest.importorskip("finufft")
    example, run = tmp_path/"example", tmp_path/"run"
    code, result, _ = call(capsys, "example", tool, "--output", example)
    assert code == 0, result
    manifest = example/"input.json"
    # Resource paths must resolve against the manifest, not the process cwd.
    monkeypatch.chdir(tmp_path)
    code, result, _ = call(capsys, "validate", tool, manifest)
    assert code == 0 and "validation_scope" in result["result"], result
    extra = ["--particles", "32", "--sweeps", "1", "--stop-after", "1"] if tool == "fit_sky" else []
    code, result, progress = call(capsys, "run", tool, manifest, "--output", run, *extra)
    assert code == 0 and result["ok"] is True, result
    assert result["full_objective_complete"] is False
    assert Path(result["artifacts"]["report.json"]).is_file()
    assert agent_cli.read(run/"provenance.json")["manifest_sha256"] == agent_cli.digest(manifest)
    if tool == "fit_sky":
        assert result["scientific_status"] == "approximate_uncertified"
        assert json.loads(progress)["event"] == "exposure_complete"
    if tool == "kepler-fold":
        assert agent_cli.read(run/"report.json")["validation"]["identical_top_and_detections"]
    if tool == "kepler-chart":
        assert result["result"]["full_score_error_certified"] is False
    before = agent_cli.digest(run/"report.json")
    code, result, _ = call(capsys, "run", tool, manifest, "--output", run)
    assert code == 2 and "already exists" in result["error"]["message"]
    assert agent_cli.digest(run/"report.json") == before
    assert not (run/"failure.json").exists()
    code, result, _ = call(capsys, "inspect", run)
    assert code == 0 and result["result"]["ok"] is True


def test_circular_propagation_against_analytic_solution(tmp_path, capsys):
    from magictools.orbits import GM
    radius = 3.
    speed = np.sqrt(GM["sun"]/radius)
    value = dict(schema_version=1, states=[dict(id="circle", state=[radius, 0., 0., 0., speed, 0.])], times_days=[0., 8., -8.])
    manifest = tmp_path/"input.json"
    manifest.write_text(json.dumps(value))
    code, result, _ = call(capsys, "run", "kepler-propagate", manifest, "--output", tmp_path/"out")
    assert code == 0, result
    report = agent_cli.read(tmp_path/"out/report.json")
    angle = speed/radius*np.array(value["times_days"])
    expected = np.column_stack([radius*np.cos(angle), radius*np.sin(angle), np.zeros(3),
                                -speed*np.sin(angle), speed*np.cos(angle), np.zeros(3)])
    assert np.array(report["states"])[0] == pytest.approx(expected, abs=1e-12)
    assert report["ra_dec_deg"] is None


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":1}',
                                  '{"value":NaN}', '{"value":1e400}', '{"unknown":1}'])
def test_malformed_inputs_are_json_errors_without_output(text, tmp_path, capsys):
    manifest = tmp_path/"bad.json"
    manifest.write_text(text)
    code, result, _ = call(capsys, "run", "kepler-propagate", manifest, "--output", tmp_path/"out")
    assert code == 2 and result["ok"] is False
    assert not (tmp_path/"out").exists()


def test_missing_dependency_is_actionable(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(agent_cli, "dependencies", lambda _: {"flint": False})
    code, result, _ = call(capsys, "example", "fit_sky", "--output", tmp_path/"out")
    assert code == 3 and "pip install" in result["error"]["message"]
    assert not (tmp_path/"out").exists()


def test_fitsky_interruption_inspection_and_resume(tmp_path, capsys, monkeypatch):
    pytest.importorskip("flint")
    from research.fit_sky.sequential import SequentialFit
    example, run = tmp_path/"example", tmp_path/"run"
    assert call(capsys, "example", "fit_sky", "--output", example)[0] == 0
    manifest = example/"input.json"
    original = SequentialFit.assimilate
    def interrupted(self, endpoint=None, **kwargs):
        if self.assimilated == 1:
            raise KeyboardInterrupt()
        return original(self, endpoint, **kwargs)
    monkeypatch.setattr(SequentialFit, "assimilate", interrupted)
    code, result, _ = call(capsys, "run", "fit_sky", manifest, "--output", run,
                           "--particles", "32", "--sweeps", "1")
    assert code == 130 and result["resume_available"]
    assert agent_cli.read(run/"failure.json")["ok"] is False
    code, result, _ = call(capsys, "inspect", run/"checkpoint.npz")
    assert code == 0 and result["result"]["assimilated"] == 1
    monkeypatch.setattr(SequentialFit, "assimilate", original)
    code, result, _ = call(capsys, "run", "fit_sky", manifest, "--output", tmp_path/"resume",
                           "--resume", run/"checkpoint.npz", "--catalogues")
    assert code == 0 and result["result"]["assimilated_exposures"] == 4
    assert agent_cli.read(tmp_path/"resume/report.json")["accuracy_target_met"] is False
    assert "catalogues" in agent_cli.read(tmp_path/"resume/report.json")


def test_wrong_engine_options_and_unknown_scene_fields(tmp_path, capsys):
    pytest.importorskip("flint")
    assert call(capsys, "example", "kepler-propagate", "--output", tmp_path/"example")[0] == 0
    code, result, _ = call(capsys, "run", "kepler-propagate", tmp_path/"example/input.json",
                           "--output", tmp_path/"out", "--particles", "2")
    assert code == 2 and "FitSky" in result["error"]["message"]
    assert not (tmp_path/"out").exists()
    assert call(capsys, "example", "fit_sky", "--output", tmp_path/"fit")[0] == 0
    manifest = tmp_path/"fit/input.json"
    value = agent_cli.read(manifest)
    value["scene"]["exposures"][0]["observer_position_typo"] = [1., 0., 0.]
    manifest.write_text(json.dumps(value))
    code, result, _ = call(capsys, "validate", "fit_sky", manifest)
    assert code == 2 and "observer_position_typo" in result["error"]["message"]


def test_invalid_budgets_and_complex_geometry_fail_admission(tmp_path, capsys):
    for option, value in [("--particles", 1), ("--pixel-block", 0), ("--seed", -1)]:
        code, result, _ = call(capsys, "run", "fit_sky", "unused.json", "--output", tmp_path/"out", option, value)
        assert code == 2 and "at least" in result["error"]["message"]
        assert not (tmp_path/"out").exists()
    pytest.importorskip("flint")
    pytest.importorskip("finufft")
    assert call(capsys, "example", "kepler-chart", "--output", tmp_path/"chart")[0] == 0
    path = tmp_path/"chart/fields.npz"
    with np.load(path) as a:
        arrays = {k: a[k].copy() for k in a.files}
    arrays["frequencies"] = arrays["frequencies"].astype(complex)+1j
    np.savez(path, **arrays)
    code, result, _ = call(capsys, "validate", "kepler-chart", tmp_path/"chart/input.json")
    assert code == 2 and "Real-valued" in result["error"]["message"]


def test_legacy_fitsky_science_id_resolves_to_canonical_schema(capsys):
    code, result, _ = call(capsys, "schema", "fitsky")
    assert code == 0 and result["tool"] == "fit_sky"
    assert result["result"] == schema("fit_sky") == schema("fitsky")


def test_fitsky_direct_and_agent_share_checkpoints_and_results(tmp_path, capsys):
    pytest.importorskip("flint")
    def science(value):
        if isinstance(value, dict):
            return {k: science(v) for k, v in value.items() if k != "seconds"}
        if isinstance(value, list):
            return [science(v) for v in value]
        return value
    example = tmp_path / "example"
    assert call(capsys, "example", "fit_sky", "--output", example)[0] == 0
    manifest = example / "input.json"
    direct, agent, resumed = tmp_path / "direct", tmp_path / "agent", tmp_path / "resumed"
    options = ["--particles", "32", "--sweeps", "1", "--seed", "7", "--stop-after", "1"]
    assert agent_cli.main(["fit_sky", str(manifest), str(direct), *options]) == 0
    direct_progress = json.loads(capsys.readouterr().out)
    code, result, progress = call(capsys, "run", "fitsky", manifest, "--output", agent, *options)
    assert code == 0 and result["tool"] == "fit_sky"
    assert science(json.loads(progress)["summary"]) == science(direct_progress)
    assert science(json.loads((direct / "report.json").read_text())) == science(json.loads((agent / "report.json").read_text()))
    with np.load(direct / "checkpoint.npz", allow_pickle=False) as a, np.load(agent / "checkpoint.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if key == "metadata":
                assert science(json.loads(str(a[key]))) == science(json.loads(str(b[key])))
            else:
                np.testing.assert_array_equal(a[key], b[key])
    assert agent_cli.main(["fit-sky-stream", str(manifest), str(resumed),
                           "--resume", str(agent / "checkpoint.npz"), "--stop-after", "2"]) == 0
    capsys.readouterr()
    code, result, _ = call(capsys, "inspect", resumed / "checkpoint.npz")
    assert code == 0 and result["result"]["assimilated"] == 2
