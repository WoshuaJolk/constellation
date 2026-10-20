"""Exercise actual observer records and independent pixel statistics."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("finufft")

from research.kepler_transform.detector_time.real_time import EmpiricalObserverTimePlan, errors
from research.kepler_transform.detector_time.scaling import real


@pytest.fixture(scope="module")
def context():
    return real(96, Path(__file__).resolve().parents[1] /
                "research/kepler_transform/chart_results/real_inputs")


@pytest.mark.parametrize("order,tolerance", [(2, 1e-4), (4, 1e-8)])
def test_real_observer_time_representation_matches_pixels(context, order, tolerance):
    plan = EmpiricalObserverTimePlan(context, order)
    positions = context.positions(context.queries)
    np.testing.assert_allclose(plan.positions(context.queries, context.times), positions, atol=1e-9, rtol=0)
    n, i = context.detector.direct(context.images, positions)
    evaluator = plan.query_plan(context.queries)
    an, ai = evaluator.apply(plan.transform(context.gn)), evaluator.apply(plan.transform(context.gi))
    assert errors(an, ai, n, i)["max_snr_error"] < tolerance
    changed = context.images * .4 + np.random.default_rng(50).normal(size=context.images.shape)*.1
    expected, _ = context.detector.direct(changed, positions)
    actual = evaluator.apply(plan.transform(context.prepare(changed)))
    assert np.max(abs(actual-expected)) < tolerance


def test_empirical_plan_rejects_outside_domain_and_mismatched_fields(context):
    for order in [True, 1, 65, 2.5]:
        with pytest.raises(ValueError):
            EmpiricalObserverTimePlan(context, order)
    plan = EmpiricalObserverTimePlan(context, 8)
    query = context.queries[:1].copy()
    query[0, 0] = np.nextafter(plan.upper[0], np.inf)
    with pytest.raises(ValueError):
        plan.query_plan(query)
    with pytest.raises(ValueError):
        plan.positions(context.queries, [context.times[-1]+1.])
    with pytest.raises(ValueError):
        plan.transform(context.gn[:2])
