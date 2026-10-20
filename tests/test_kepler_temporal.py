"""Time compression: domain refusal, independent geometry, Fourier and adjoint checks."""

from itertools import product

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("numba")

from research.kepler_transform.temporal.transform import (
    FourierEvaluator, TimePlan, certificate, image_coefficients,
)


@pytest.fixture(scope="module")
def workload():
    rng = np.random.default_rng(421)
    # Irregular cadence, full eight-day span, unchanged six-dimensional box.
    times = np.r_[-4., np.sort(rng.uniform(-4, 4, 103)), 4.]
    plan = TimePlan("near", times, (5, 6))
    q = np.r_[np.array(list(product([-1., 1.], repeat=6))),
              rng.uniform(-1, 1, (32, 6))]
    data = rng.normal(size=(len(times), 5, 6)) + 1j * rng.normal(size=(len(times), 5, 6))
    return plan, q, data


def test_full_box_scores_and_independent_geometry(workload):
    plan, q, data = workload
    field = plan.transform(data)
    actual = plan.evaluate(field, q, backend="stream")
    direct = plan.direct_plan(q, backend="stream").apply(data)
    assert max(abs(actual - direct)) / sum(abs(data.ravel())) < plan.proof["exact_arithmetic_kernel_error_bound"]
    # Independent Python universal-variable/light-time solver, both interval endpoints.
    sources = np.array([[-1., -1., 1.], [1., 1., -1.], [.123, .4, -.7]])
    assert np.max(abs(plan.phase(q[:8], sources) - plan.phase.independent(q[:8], sources))) < 2e-8
    assert plan.order == 82
    assert plan.counts(len(q))["orbit_grid_entries"] == 0


def test_backends_and_adjoint(workload):
    pytest.importorskip("finufft")
    plan, q, data = workload
    rng = np.random.default_rng(320)
    q = q[:12]
    fields = plan.transform(data)
    dense = plan.query_plan(q, backend="dense")
    stream = plan.query_plan(q, backend="stream")
    nufft = plan.query_plan(q, backend="nufft")
    reference = dense.apply(fields)
    np.testing.assert_allclose(stream.apply(fields), reference, atol=2e-10, rtol=2e-11)
    np.testing.assert_allclose(nufft.apply(fields), reference, atol=2e-8, rtol=2e-10)
    y = rng.normal(size=len(q)) + 1j * rng.normal(size=len(q))
    for evaluator in (dense, stream):
        left = np.vdot(evaluator.apply(fields), y)
        right = np.vdot(data, plan.adjoint_compress(evaluator.adjoint(y)))
        np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-12)


def test_fft_pixel_input_convention():
    rng = np.random.default_rng(44)
    shape = (5, 6)
    images = rng.normal(size=(3, *shape))
    coefficients = image_coefficients(images)
    xx, yy = np.meshgrid(2 * np.pi * np.arange(shape[0]) / shape[0],
                         2 * np.pi * np.arange(shape[1]) / shape[1], indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    # Evaluate one field at a time at its original pixel positions.
    for i in range(3):
        evaluator = FourierEvaluator(xy[:, None], shape, [1., 1.], backend="dense")
        np.testing.assert_allclose(evaluator.apply(coefficients[i:i + 1]).real,
                                   images[i].ravel(), atol=3e-15, rtol=2e-14)


def test_zero_single_source_and_linear_superposition(workload):
    plan, q, data = workload
    q = q[:6]
    evaluator = plan.query_plan(q, backend="dense")
    zeros = np.zeros_like(data)
    np.testing.assert_array_equal(evaluator.apply(plan.transform(zeros)), 0)
    # Isolate a boundary-frequency atom in the last exposure.
    atom = zeros.copy()
    atom[-1, 0, 0] = 1
    result = evaluator.apply(plan.transform(atom))
    reference = plan.direct_plan(q, backend="stream").apply(atom)
    assert max(abs(result - reference)) < plan.proof["exact_arithmetic_kernel_error_bound"]
    np.testing.assert_allclose(evaluator.apply(plan.transform(data + 2j * atom)),
                               evaluator.apply(plan.transform(data)) + 2j * result,
                               atol=3e-11, rtol=3e-12)


def test_domain_and_resource_refusal(workload):
    plan, q, data = workload
    for invalid in (np.full((1, 6), 1.001), np.full((1, 6), np.nan), np.zeros((1, 5))):
        with pytest.raises(ValueError):
            plan.query_plan(invalid)
    with pytest.raises(ValueError, match="budget"):
        plan.query_plan(q, backend="dense", max_cache_bytes=1)
    with pytest.raises(ValueError, match="interval"):
        TimePlan("near", [-4., 4.001])
    with pytest.raises(ValueError, match="order"):
        certificate("near", max_order=2)
    with pytest.raises(ValueError):
        plan.transform(data[:-1])
    with pytest.raises(ValueError):
        image_coefficients(np.array([np.nan]))


def test_distant_and_stress_entire_domain_certificates():
    for case, rho, expected in [("distant", 2., 127), ("stress", 1.5, 490)]:
        proof = certificate(case, rho=rho)
        assert proof["order"] == expected
        assert proof["exact_arithmetic_kernel_error_bound"] <= 1e-8
        assert proof["root_certificate"]["certified"]
        assert not proof["floating_point_error_certified"]


def test_fast_time_moments_and_adjoint(workload):
    pytest.importorskip("finufft")
    dense, _, data = workload
    fast = TimePlan("near", dense.times, dense.shape, compression_backend="nufft")
    np.testing.assert_allclose(fast.transform(data), dense.transform(data), atol=2e-9, rtol=2e-10)
    rng = np.random.default_rng(642)
    v = rng.normal(size=(fast.order, *fast.shape)) + 1j * rng.normal(size=(fast.order, *fast.shape))
    np.testing.assert_allclose(np.vdot(fast.transform(data), v),
                               np.vdot(data, fast.adjoint_compress(v)), atol=2e-8, rtol=2e-10)
    assert fast.counts(10)["interpolation_real_entries"] == 0
