"""Independent ODE and finite-difference checks across year-separated epochs."""
import numpy as np
import os
import pytest
from scipy.integrate import solve_ivp

from magictools.itf.nbody import SUN_MU, Trajectory, acceleration, covariance_factor, fit_nbody
from magictools.itf.kernels import C


class CircularJupiter:
    start, end = -5000., 5000.
    min_mjd, max_mjd = start, end
    masses = np.array([SUN_MU / 1047.35])
    sun_mu = SUN_MU

    def positions(self, t):
        angle = np.asarray(t) * np.sqrt(SUN_MU / 5.2**3)
        return np.stack((5.2*np.cos(angle), 5.2*np.sin(angle), np.zeros_like(angle)), axis=-1)[..., None, :]

    def earth(self, t):
        angle = np.asarray(t)*np.sqrt(SUN_MU)
        return np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=-1)


def independent_state(state, times, ephem):
    def rhs(t, y):
        planet = np.array([5.2*np.cos(t*np.sqrt(SUN_MU/5.2**3)),
                           5.2*np.sin(t*np.sqrt(SUN_MU/5.2**3)), 0.])
        offset = planet-y[:3]
        force = -SUN_MU*y[:3]/np.linalg.norm(y[:3])**3
        force += ephem.masses[0]*(offset/np.linalg.norm(offset)**3-planet/np.linalg.norm(planet)**3)
        return np.concatenate((y[3:], force))
    answer = np.empty((len(times), 6))
    for sign in (-1, 1):
        use = np.flatnonzero(times*sign >= 0)
        target = sign*max(abs(times[use]).max(), 1e-9)
        solution = solve_ivp(rhs, (0, target), state, method='DOP853', rtol=3e-13, atol=2e-14, dense_output=True)
        answer[use] = solution.sol(times[use]).T
    return answer


def test_planetary_force_jacobian():
    ephem = CircularJupiter()
    position = np.array([2., .4, .1])
    _, gradient = acceleration(position, ephem.positions(100), ephem.masses)
    for j in range(3):
        delta = np.eye(3)[j]*1e-5
        finite = (acceleration(position+delta, ephem.positions(100), ephem.masses)[0] -
                  acceleration(position-delta, ephem.positions(100), ephem.masses)[0])/(2e-5)
        np.testing.assert_allclose(gradient[:, j], finite, rtol=1e-8, atol=1e-12)


def test_planetary_four_observation_training_fit_requires_explicit_opt_in():
    from types import SimpleNamespace
    ephem=CircularJupiter();state=np.array([2.2,.2,.15,-.002,.0105,.001])
    times=np.array([-2.,-1.98,3.,3.02]);observers=ephem.earth(times);emitted=times.copy()
    for _ in range(6):
        xyz=independent_state(state,emitted,ephem)[:,:3]-observers
        emitted=times-np.linalg.norm(xyz,axis=1)/C
    unit=xyz/np.linalg.norm(xyz,axis=1)[:,None]
    geometry=SimpleNamespace(ephemeris=ephem,get=lambda ids:(times,unit,observers,np.array([0,0,1,1])))
    assert fit_nbody(geometry,[0,1],state,0) is None
    fit=fit_nbody(geometry,[0,1],state+np.array([1e-5,0,0,0,0,0]),0,allow_two_tracklets=True)
    assert fit is not None and fit['observations']==4
    assert fit['rms_arcsec']<.01


def test_year_propagation_and_lighttime_jacobian_against_independent_ode():
    ephem = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    times = np.array([-730., -365., 365., 730.])
    trajectory = Trajectory(state, 0, -750, 750, ephem)
    actual, matrices = trajectory.evaluate(times)
    expected = independent_state(state, times, ephem)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-8)
    observers = ephem.earth(times)
    unit, distance, jacobian = trajectory.apparent(times, observers)
    emitted = times.copy()
    for _ in range(6):
        position = independent_state(state, emitted, ephem)[:, :3] - observers
        emitted = times-np.linalg.norm(position, axis=1)/C
    np.testing.assert_allclose(unit, position/np.linalg.norm(position, axis=1)[:, None], atol=2e-8)
    for j in range(6):
        h = 1e-6 if j < 3 else 1e-8
        delta = np.eye(6)[j]*h
        plus = Trajectory(state+delta, 0, -750, 750, ephem).apparent(times, observers)[0]
        minus = Trajectory(state-delta, 0, -750, 750, ephem).apparent(times, observers)[0]
        np.testing.assert_allclose(jacobian[:, :, j], (plus-minus)/(2*h), rtol=2e-5, atol=2e-5)


def test_joint_fit_recovers_before_and_after_year_arcs_and_rejects_interloper():
    ephem = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    times = np.array([-365., -364.98, 0., .02, 2., 2.02, 5., 5.02, 370., 370.02])
    observers = ephem.earth(times)
    emitted = times.copy()
    for _ in range(6):
        position = independent_state(state, emitted, ephem)[:, :3] - observers
        emitted = times - np.linalg.norm(position, axis=1)/C
    observed = position/np.linalg.norm(position, axis=1)[:, None]
    labels = np.repeat(np.arange(5), 2)

    class Geometry:
        ephemeris = ephem

        def get(self, indices):
            selected = np.isin(labels, indices)
            return times[selected], observed[selected], observers[selected], labels[selected]

    initial = state + np.array([1e-4, -2e-4, 1e-4, 1e-6, -1e-6, 1e-6])
    result = fit_nbody(Geometry(), list(range(5)), initial, 0)
    assert result is not None and result['span_days'] > 730
    assert result['rms_arcsec'] < .01
    np.testing.assert_allclose(result['state_au_au_per_day'], state, atol=2e-7)
    from magictools.itf.association_likelihood import NightCovariance
    errors = NightCovariance(np.tile(np.eye(2)*.09,(len(times),1,1)),labels,np.eye(2)*.04)
    weighted = fit_nbody(Geometry(),list(range(5)),initial,0,observation_covariance=errors)
    assert weighted is not None
    np.testing.assert_allclose(weighted['state_au_au_per_day'], state, atol=2e-7)
    assert weighted['astrometric_error_model']['calibrated'] is False
    # With a zero shared offset and the original random sigma, recover the
    # existing fit and covariance, not merely the same acceptance outcome.
    original_errors=NightCovariance(np.tile(np.eye(2)*.09,(len(times),1,1)),labels,np.zeros((2,2)))
    equivalent=fit_nbody(Geometry(),list(range(5)),initial,0,observation_covariance=original_errors)
    f=np.asarray(equivalent['covariance_factor']);g=np.asarray(result['covariance_factor'])
    np.testing.assert_allclose(f@f.T,g@g.T,rtol=2e-5,atol=1e-13)
    observed[-2:, 0] += .01
    observed[-2:] /= np.linalg.norm(observed[-2:], axis=1)[:, None]
    assert fit_nbody(Geometry(), list(range(5)), initial, 0) is None


def test_covariance_factor_matches_independent_inverse():
    rng = np.random.default_rng(104)
    jacobian = rng.normal(size=(30, 6))*np.array([1, 2, 3, 10, 20, 30])
    factor = covariance_factor(jacobian)
    np.testing.assert_allclose(factor@factor.T, np.linalg.inv(jacobian.T@jacobian), atol=1e-13)


@pytest.mark.skipif(not os.environ.get('ITF_TEST_EPHEMERIS'), reason='Optional local JPL SPK')
def test_jpl_force_interpolation_and_exact_observer():
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    from astropy import units
    from magictools.itf.nbody import Ephemeris, AU_KM
    path = os.environ['ITF_TEST_EPHEMERIS']
    ephemeris = Ephemeris(path, 59990, 60020)
    times = np.array([60000.12, 60001.29, 60003.77])
    positions, _ = ephemeris.exact(times)
    assert np.max(np.linalg.norm(positions-ephemeris.positions(times), axis=2))*AU_KM < .3
    t = Time(times, format='mjd', scale='tdb')
    earth = get_body_barycentric('earth', t, ephemeris=path)
    sun = get_body_barycentric('sun', t, ephemeris=path)
    np.testing.assert_allclose(ephemeris.earth(times), (earth.xyz-sun.xyz).to_value(units.au).T, atol=1e-10)
