"""Compiled integration against SciPy's independently executed adaptive solver."""
import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from magictools.itf.fit import BudgetExpired
from magictools.itf.nbody import SUN_MU, Trajectory, acceleration
from magictools.itf.nbody_fast import derivative


class SplineJupiter:
    start, end = -2200., 2200.
    masses = np.array([SUN_MU/1047.35])
    sun_mu = SUN_MU

    def __init__(self):
        t = np.arange(self.start, self.end+.5, .5)
        rate = np.sqrt(SUN_MU/5.2**3)
        p = np.stack((5.2*np.cos(rate*t), 5.2*np.sin(rate*t), np.zeros(len(t))), axis=-1)
        v = np.stack((-5.2*rate*np.sin(rate*t), 5.2*rate*np.cos(rate*t), np.zeros(len(t))), axis=-1)
        self.spline = CubicHermiteSpline(t, p[:, None], v[:, None])

    def positions(self, t):
        return self.spline(t)


def test_compiled_force_and_variational_derivative():
    ephem = SplineJupiter()
    rng = np.random.default_rng(27)
    for t in (ephem.start, -.123, 700., ephem.end):
        y = np.r_[np.array([2., .2, .1, -.001, .01, 0.]), rng.normal(size=36)]
        result = derivative(t, y, ephem.spline.x, ephem.spline.c, ephem.masses, SUN_MU)
        acc, grad = acceleration(y[:3], ephem.positions(t), ephem.masses)
        matrix = y[6:].reshape(6, 6)
        expected = np.r_[y[3:6], acc, np.vstack((matrix[3:], grad@matrix[:3])).ravel()]
        np.testing.assert_allclose(result, expected, rtol=5e-13, atol=1e-15)


@pytest.mark.parametrize('state', [
    [2.2, .2, .15, -.002, .0105, .001],
    [2.2, .2, .15, .002, -.0105, -.001],  # retrograde
    [1.4, .2, .15, -.002, .025, .002],  # unbound
    [30., .2, 2., -.001, .002, .001],
])
@pytest.mark.parametrize('backend', ['compiled', 'scipy_compiled_rhs'])
def test_compiled_bidirectional_state_stm_and_lighttime(state, backend):
    ephem = SplineJupiter()
    times = np.r_[np.linspace(-1900, 1900, 301), 0., 1900., -1900.]
    reference = Trajectory(state, 0., -2000., 2000., ephem, backend='scipy')
    compiled = Trajectory(state, 0., -2000., 2000., ephem, backend=backend)
    actual, matrix = compiled.evaluate(times)
    expected, expected_matrix = reference.evaluate(times)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(matrix, expected_matrix, rtol=2e-7, atol=2e-5)
    observers = np.column_stack((np.cos(times*.017), np.sin(times*.017), np.zeros(len(times))))
    for result, truth in zip(compiled.apparent(times, observers), reference.apparent(times, observers)):
        np.testing.assert_allclose(result, truth, rtol=3e-7, atol=2e-7)


@pytest.mark.parametrize('backend', ['compiled', 'scipy_compiled_rhs'])
def test_compiled_deadline_and_solar_collision(backend):
    ephem = SplineJupiter()
    with pytest.raises(BudgetExpired):
        Trajectory([2., 0., 0., 0., .01, 0.], 0., -1., 1., ephem, backend=backend, deadline=0.)
    with pytest.raises(ValueError, match='Sun'):
        Trajectory([.001, 0., 0., 0., .01, 0.], 0., -1., 1., ephem, backend=backend)


def test_segment_extension_carries_original_epoch_covariance():
    ephem = SplineJupiter()
    state = [2.2, .2, .15, -.002, .0105, .001]
    segmented = Trajectory(state, 0., -10., 10., ephem, backend='compiled')
    original_segments = list(segmented.segments)
    segmented.extend(-700., 900.)
    segmented.extend(-1900., 1900.)
    assert all(segment is saved for segment, saved in zip(segmented.segments, original_segments))
    reference = Trajectory(state, 0., -1900., 1900., ephem, backend='scipy')
    times = np.array([-1900., -700., -10., 0., 10., 900., 1900.])
    for actual, expected in zip(segmented.evaluate(times), reference.evaluate(times)):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-5)


def test_fast_mode_light_time_extends_beyond_fixed_65_day_margin():
    from scipy.optimize import brentq
    from magictools.itf.kernels import C
    ephem = SplineJupiter()
    ephem.sun_mu = 0.
    ephem.masses = np.zeros_like(ephem.masses)
    state = np.array([30., 20., .1, 8., 2., 0.])
    times = np.array([-1900., -500., 100.])
    observers = np.tile([1., 0., 0.], (len(times), 1))
    trajectory = Trajectory(state, 0., -1900.05, 100.05, ephem, backend='compiled')
    actual = trajectory.apparent(times, observers)[0]
    emitted = np.array([brentq(lambda t: t-receive+np.linalg.norm(state[:3]+state[3:]*t-observer)/C,
        receive-200, receive) for receive, observer in zip(times, observers)])
    ray = state[:3] + emitted[:, None]*state[3:] - observers
    np.testing.assert_allclose(actual, ray/np.linalg.norm(ray, axis=1)[:, None], atol=2e-12)
    assert trajectory.start < -1965.
