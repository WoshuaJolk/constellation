"""Common angular-grid fields against independent nonperiodic pixel integrals."""

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("finufft")

from research.kepler_transform.detector_time.fields import CommonDetector, l1_upper, trim_fields
from research.kepler_transform.temporal.transform import FourierEvaluator


def fixture_detector(spectral_sigma=8.):
    rng = np.random.default_rng(881)
    images = rng.normal(size=(3, 7, 7))
    variances = rng.uniform(.5, 2., images.shape)
    matrices = np.array([[[1., .1], [.2, .9]], [[.93, -.3], [.3, 1.02]],
                         [[-1.03, .02], [.13, 1.05]]])
    masks = rng.uniform(size=images.shape) < .1
    detector = CommonDetector(variances, [[.8, 1.2], [1.4, .9], [1., 1.]],
                              matrices, offsets=[[.2, -.3], [-.4, .1], [.6, -.2]],
                              masks=masks, flux_scales=[.8, 1., 1.2],
                              source_half_width=4., spectral_sigma=spectral_sigma)
    p = rng.uniform(-4, 4, (80, 3, 2))
    p[:4, :, :] = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]])[:, None]
    return detector, images, p


def test_affine_fields_match_finite_pixels_at_boundaries():
    detector, images, p = fixture_detector()
    n, i, report = detector.prepare(images)
    evaluator = FourierEvaluator(p, n.shape[1:], [detector.spacing] * 2, backend="nufft")
    actual = [evaluator.apply(x) for x in (n, i)]
    direct = detector.direct(images, p)
    for k, name in enumerate(["numerator", "information"]):
        bound = sum(r[f"{name}_model_error_bound"] for r in report["exposures"])
        assert max(abs(actual[k] - direct[k])) <= bound + 2e-11
        assert max(abs(actual[k].imag)) < 1e-11
    assert not report["periodic_camera_assumed"]
    assert not report["image_resampling_performed"]


def test_loose_tail_and_trimming_are_charged():
    detector, images, p = fixture_detector(3.)
    n, i, report = detector.prepare(images)
    evaluator = FourierEvaluator(p, n.shape[1:], [detector.spacing] * 2, backend="nufft")
    direct = detector.direct(images, p)
    for actual, reference, key in zip([evaluator.apply(n), evaluator.apply(i)], direct,
                                       ["numerator", "information"]):
        bound = sum(r[f"{key}_model_error_bound"] for r in report["exposures"])
        assert max(abs(actual - reference)) <= bound + 1e-11
    a, b, trim = trim_fields(n, i, [1e-3, 2e-3])
    small = FourierEvaluator(p, a.shape[1:], [detector.spacing] * 2, backend="nufft")
    for full, cut, bound in [(n, a, trim["discarded_numerator_l1_bound"]),
                             (i, b, trim["discarded_information_l1_bound"])]:
        assert max(abs(evaluator.apply(full) - small.apply(cut))) <= bound + 1e-11
    assert a.shape == b.shape


def test_masked_pixels_do_not_contribute():
    detector = CommonDetector(np.ones((1, 5, 5)), [1., 1.], np.eye(2),
                              masks=np.ones((1, 5, 5)), source_half_width=3.)
    n, i, _ = detector.prepare(np.full((1, 5, 5), 1e6))
    np.testing.assert_array_equal(n, 0.)
    np.testing.assert_array_equal(i, 0.)


def test_error_bound_and_input_refusals():
    x = np.array([1. + 2j, 3. - 4j, 1e-20])
    assert l1_upper(x) >= sum(abs(x))
    detector, images, p = fixture_detector()
    with pytest.raises(ValueError, match="box"):
        detector.direct(images, p * 2)
    with pytest.raises(ValueError, match="allocation"):
        CommonDetector(np.ones((1, 5, 5)), [1., 1.], np.eye(2), max_bytes=1)
    with pytest.raises(ValueError):
        CommonDetector(np.ones((1, 5, 5)), [1., 1.], np.zeros((2, 2)))


def test_pixel_caches_and_numerator_only_preparation():
    from research.kepler_transform.detector_time.pixel_cache import PixelCache
    detector, images, p = fixture_detector()
    for backend in ("factor", "dense", "stream"):
        cache = PixelCache(detector, p, backend=backend)
        for data in [images, images * .3 + 2.]:
            actual = cache.apply(data)
            reference = detector.direct(data, p)
            np.testing.assert_allclose(actual, reference, atol=3e-15, rtol=1e-12)
        with pytest.raises(ValueError, match="budget"):
            PixelCache(detector, p, backend=backend, max_bytes=1)
    full, _, _ = detector.prepare(images)
    n, i, report = detector.prepare(images, include_information=False)
    assert i is None and not report["information_computed"]
    np.testing.assert_allclose(n, full, atol=2e-13, rtol=2e-11)


def test_actual_bandwidth_plan_and_statistic_intervals():
    from research.kepler_transform.detector_time.plan import (
        DetectorTimePlan, domain_source_box, statistic_intervals)
    from research.kepler_transform.factorization import ARCSEC
    rng = np.random.default_rng(231)
    plan = DetectorTimePlan("near", [-4., -.1, 1., 4.], (9, 9), .8)
    assert plan.proof["frequency_bound_radian_inverse"] > 3 / ARCSEC
    assert plan.order > 82
    data = rng.normal(size=(4, 9, 9)) + 1j * rng.normal(size=(4, 9, 9))
    q = rng.uniform(-1, 1, (12, 6))
    actual = plan.evaluate(plan.transform(data), q, backend="nufft")
    reference = plan.direct_plan(q, backend="nufft").apply(data)
    assert max(abs(actual - reference)) < plan.score_error_bound(data) + 1e-10
    bounds = domain_source_box("near")
    positions = (plan.positions(q, plan.times) - plan.center_input) / ARCSEC
    assert np.all(abs(positions) <= np.array(bounds["relative_reference_position_bound_arcsec"]))
    intervals = statistic_intervals([1., -1., 1.], [2., 2., 0.], .1, .1)
    assert intervals["positive_information_resolved"].tolist() == [True, True, False]
    assert np.isneginf(intervals["snr_interval"][2, 0])


def test_fast_compression_beyond_previous_order_cap():
    from research.kepler_transform.detector_time.plan import DetectorTimePlan
    rng = np.random.default_rng(901)
    plan = DetectorTimePlan("near", [-4., 0., 4.], (5, 5), 6., compression_backend="nufft")
    assert plan.order > 512
    data = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(size=(3, 5, 5))
    q = rng.uniform(-1, 1, (3, 6))
    actual = plan.evaluate(plan.transform(data), q, backend="nufft")
    direct = plan.direct_plan(q, backend="stream").apply(data)
    assert max(abs(actual - direct)) <= plan.score_error_bound(data) + 1e-10
    with pytest.raises(ValueError, match="budget"):
        DetectorTimePlan("near", [-4., 4.], (5, 5), 6., max_order=4)


def test_numerator_specific_grid_is_bounded_without_full_mode_generation():
    detector, images, p = fixture_detector()
    cutoff, report = detector.choose_numerator_cutoff(images, 1e-5)
    n, actual_report = detector.prepare_numerator(images, cutoff)
    assert report == actual_report
    direct = detector.direct(images, p)[0]
    evaluator = FourierEvaluator(p, n.shape[1:], [detector.spacing] * 2, backend="nufft")
    assert max(abs(evaluator.apply(n) - direct)) <= report["numerator_model_error_bound"] + 1e-11
    assert cutoff <= detector.cutoff
