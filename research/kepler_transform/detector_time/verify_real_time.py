"""Replay real-time experiment hashes, saved errors, and fresh pixel checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .real_time import EmpiricalObserverTimePlan, errors
from .scaling import real


def verify(directory):
    root = Path(__file__).resolve().parents[3]
    report = json.loads((directory / "results.json").read_text())
    for key in ["source_sha256", "input_sha256", "artifact_sha256"]:
        base = directory if key == "artifact_sha256" else root
        for name, digest in report[key].items():
            if hashlib.sha256((base/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Hash mismatch: {name}")
    with np.load(directory / "scores.npz") as archive:
        a = {name: archive[name] for name in archive.files}
    q, dn, di = a["queries"], a["direct_n"], a["direct_i"]
    assert len(q) == report["queries"]
    for row in report["orders"]:
        actual = errors(a[f"n_{row['order']}"], a[f"i_{row['order']}"], dn, di)
        for key, value in actual.items():
            assert np.isclose(value, row[key], atol=1e-15, rtol=1e-10)
    ctx = real(len(q), root / "research/kepler_transform/chart_results/real_inputs")
    np.testing.assert_array_equal(ctx.queries, q)
    indices = np.unique(np.r_[np.arange(65), np.linspace(0, len(q)-1, 32, dtype=int)])
    expected_n, expected_i = ctx.detector.direct(ctx.images, ctx.positions(q[indices]))
    np.testing.assert_allclose(expected_n, dn[indices], atol=1e-8, rtol=0)
    np.testing.assert_allclose(expected_i, di[indices], atol=1e-8, rtol=0)
    for order, prefix, target in [
        (report["selected_order"], "fast", report["threshold_max_absolute_snr_error"]),
        (report["compact"]["order"], "compact", report["compact"]["threshold_max_absolute_snr_error"]),
    ]:
        plan = EmpiricalObserverTimePlan(ctx, order)
        evaluator = plan.query_plan(q[indices])
        n, i = evaluator.apply(plan.transform(ctx.gn)), evaluator.apply(plan.transform(ctx.gi))
        assert errors(n, i, expected_n, expected_i)["max_snr_error"] < target
        np.testing.assert_allclose(n, a[f"n_{order}"][indices], atol=1e-9, rtol=0)
        np.testing.assert_allclose(i, a[f"i_{order}"][indices], atol=1e-9, rtol=0)
        holdout_error = errors(a[f"holdout_{prefix}_n"], a[f"holdout_{prefix}_i"],
                               a["holdout_direct_n"], a["holdout_direct_i"])
        assert holdout_error["max_snr_error"] < target
        changed_error = errors(a[f"changed_{prefix}_n"], a[f"i_{order}"],
                               a["changed_direct_n"], di)
        assert changed_error["max_snr_error"] < target
    return {
        "kind": "real_empirical_time",
        "order_score_pairs_replayed": len(report["orders"]),
        "queries_per_pair": len(q), "holdout_queries": len(a["holdout_queries"]),
        "independent_pixel_queries_recomputed": len(indices),
        "fresh_time_orders_checked": [report["selected_order"], report["compact"]["order"]],
        "source_input_and_artifact_hashes_checked": True,
        "whole_domain_time_certificate": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)), flush=True)
