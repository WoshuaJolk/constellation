from itertools import combinations
import json
import subprocess
import sys

import numpy as np
import pytest

from magictools.ejector import run_ejector, scan_convergences
from magictools.galactic import KMS_TO_PC_MYR


def test_pair_gating_matches_exhaustive_group_search():
    rng = np.random.default_rng(91)
    paths = np.zeros((5, 32, 7, 6))
    paths[..., :3] = rng.normal(size=(5, 32, 7, 3))
    radius, threshold = 1.4, 0.2
    found, _ = scan_convergences(paths, np.arange(5), radius, group_size=3, min_support=threshold)
    brute = {}
    for group in combinations(range(7), 3):
        points = paths[:, :, group, :3]
        residual2 = np.sum((points - points.mean(axis=2, keepdims=True))**2, axis=-1)
        support = np.mean(np.max(residual2, axis=2) <= radius**2, axis=1)
        if support.max() >= threshold:
            brute[group] = support.max()
    assert {row["indices"]: row["support"] for row in found} == brute


def test_pair_clique_is_not_mistaken_for_common_origin():
    paths = np.zeros((1, 16, 3, 6))
    paths[0, :, :, :3] = [[0, 0, 0], [1.9, 0, 0], [0.95, 1.9 * np.sqrt(3) / 2, 0]]
    found, evaluated = scan_convergences(paths, [0], 1, min_support=0.2)
    assert evaluated == 1 and found == []


def test_candidate_cap_fails_instead_of_truncating():
    with pytest.raises(ValueError, match="cap exceeded"):
        scan_convergences(np.zeros((1, 16, 8, 6)), [0], 1, max_group_evaluations=1)


@pytest.fixture
def manifest(tmp_path):
    origin = np.array([100., 200., 300.])
    velocities = np.array([[100, 0, 0], [0, 100, 0], [-50, -50, 30], [40, -30, 100]])
    stars = [{"id": f"s{i}", "state_pc_kms": np.r_[origin + 2 * v * KMS_TO_PC_MYR, v].tolist()} for i, v in enumerate(velocities)]
    stars += [{"id": "field", "state_pc_kms": [-1000, 2000, 3000, 20, -50, 40]}]
    value = {"schema_version": 1, "epoch_label": "test present", "potential": {"components": []},
             "stars": stars, "counterparts": [{"id": "cluster", "state_pc_kms": [*origin, 0, 0, 0]}],
             "search": {"lookback_max_myr": 3, "time_step_myr": 0.1, "radius_pc": 2,
                        "group_size": 4, "n_samples": 16, "n_controls": 7, "max_step_myr": 0.1}}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    return path, value


def test_recovers_unknown_members_age_origin_and_counterpart(manifest):
    path, _ = manifest
    report = run_ejector(path)
    candidate = report["candidates"][0]
    assert set(candidate["star_ids"]) == {"s0", "s1", "s2", "s3"}
    assert candidate["lookback_myr"] == pytest.approx(2)
    assert candidate["support"] == 1
    np.testing.assert_allclose(candidate["origin_given_convergence_percentiles_pc"][1], [100, 200, 300])
    assert candidate["supplied_counterpart_comparisons"][0]["conditional_proximity_fraction"] == 1
    maxima = report["controls"]["max_search_scores"]
    expected = (1 + sum(score >= candidate["score"] for score in maxima)) / 8
    assert candidate["search_wide_velocity_permutation_p"] == expected
    assert report == run_ejector(path)


def test_no_controls_does_not_invent_significance(manifest):
    path, value = manifest
    value["search"]["n_controls"] = 0
    path.write_text(json.dumps(value))
    report = run_ejector(path)
    assert report["candidates"][0]["search_wide_velocity_permutation_p"] is None
    assert report["controls"]["minimum_resolvable_p"] is None


def test_uninformative_control_strata_are_rejected(manifest):
    path, value = manifest
    for row in value["stars"]:
        row["control_stratum"] = row["id"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="singleton"):
        run_ejector(path)


def test_cli_json_and_no_overwrite(manifest, tmp_path):
    path, _ = manifest
    output = tmp_path / "result.json"
    command = [sys.executable, "-m", "magictools", "find_ejector", str(path), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    before = output.read_bytes()
    assert json.loads(before)["candidates"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert before == output.read_bytes()
