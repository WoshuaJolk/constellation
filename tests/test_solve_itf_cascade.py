import itertools
import numpy as np
import pytest

from test_solve_itf import injected_catalogue


def test_fast_atlas_retains_every_observation_and_original_order(injected_catalogue, tmp_path):
    from magictools.itf.campaign_geometry import prepare_atlas, AtlasGeometry
    from magictools.itf.campaign_geometry_fast import prepare_observation_ids, FastAtlasGeometry
    c, sites, sitefile = injected_catalogue
    atlas = tmp_path/'atlas'; identities = tmp_path/'ids'
    prepare_atlas(c.path, sitefile, atlas, lambda _: None)
    prepare_observation_ids(c.path, identities)
    slow = AtlasGeometry(c.path, sitefile, atlas)
    fast = FastAtlasGeometry(c.path, sitefile, atlas, identities)
    for indices in itertools.permutations(range(3)):
        for actual, expected in zip(fast.get(indices), slow.get(indices)):
            np.testing.assert_array_equal(actual, expected)


def test_precomputed_keys_preserve_cross_alias_dedup_and_station_distinction():
    from types import SimpleNamespace
    from magictools.itf.catalogue import OBS_DTYPE
    from magictools.itf.campaign_geometry_fast import gather
    from magictools.itf.fit import Geometry
    data = np.zeros(9, dtype=OBS_DTYPE)
    data['t'] = [1, 2, 3]*3; data['ra_mas'] = [4, 5, 6]*3
    keys = np.empty(9, dtype=OBS_DTYPE.descr+[('site', 'S3')])
    for name in OBS_DTYPE.names: keys[name] = data[name]
    keys['site'] = [b'500']*6+[b'568']*3
    _, identities = np.unique(keys, return_inverse=True)
    times = data['t']+.0001; unit = np.ones((9, 3)); observer = np.ones((9, 3))*2
    tracks = np.zeros(3, dtype=[('site', 'S3')]); tracks['site'] = [b'500', b'500', b'568']
    fake = SimpleNamespace(catalogue=SimpleNamespace(tracks=tracks),
        track=lambda i: (data[3*i:3*i+3], times[3*i:3*i+3], unit[3*i:3*i+3], observer[3*i:3*i+3]))
    for indices in itertools.permutations(range(3)):
        actual = gather(np.array(indices), np.arange(3)*3, np.full(3, 3), np.ones(3, np.uint8),
                        identities.astype(np.uint32), times, unit, observer)
        order = np.argsort(actual[0]); actual = tuple(a[order] for a in actual)
        expected = Geometry.get(fake, indices)
        assert len(actual[0]) == 6
        for a, e in zip(actual, expected): np.testing.assert_array_equal(a, e)


def test_cascade_acceptance_requires_full_observations_and_marks_deferred(injected_catalogue):
    from magictools.itf.fit import Geometry
    from magictools.itf.cascade import fit_cascade
    c, sites, _ = injected_catalogue
    row, counts = fit_cascade(Geometry(c, sites), [0, 2, 5], pilot_evaluations=12, promote_rms=100.)
    assert row is not None and row['observations'] == sum(c.tracks['count'][[0, 2, 5]])
    assert row['rms_arcsec'] <= .6 and row['full_grid_refinement_pending']
    assert counts['disposition'] == 'accepted_fit'
    row, counts = fit_cascade(Geometry(c, sites), [0, 2, 5], pilot_evaluations=1, promote_rms=0., ranges=())
    assert row is None and counts['disposition'] == 'deferred_full_grid'


@pytest.mark.parametrize('state', [
    [2.4, .7, .3, -.004, .009, .002],
    [2.4, .7, .3, .004, -.009, -.002],
    [2.4, .7, .3, -.004, .03, .002],
    [80., 30., 5., -.001, .0015, .0001],
    [1.05, .1, .02, 0., .019, .002],
])
def test_harmonic_starts_correct_to_independent_all_conic_observations(state):
    from scipy.integrate import solve_ivp
    from magictools.itf.harmonic import range_starts
    from magictools.itf.cascade import PreparedSeed
    from magictools.itf.fit import fit_chain
    from magictools.itf.kernels import MU, C
    state = np.array(state)
    offsets = np.repeat([-3., 0., 3.], 4)+np.tile([0., .01, .02, .03], 3)
    def rhs(t, y): return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
    forward = solve_ivp(rhs, [0., 5.], state, method='DOP853', rtol=2e-12, atol=1e-14, dense_output=True).sol
    backward = solve_ivp(rhs, [0., -5.], state, method='DOP853', rtol=2e-12, atol=1e-14, dense_output=True).sol
    angle = offsets*2*np.pi/365.25
    observers = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    observed = []
    for t, observer in zip(offsets, observers):
        emitted = t
        for _ in range(12):
            delta = (forward if emitted >= 0 else backward)(emitted)[:3]-observer
            emitted = t-np.linalg.norm(delta)/C
        observed.append(delta/np.linalg.norm(delta))
    observed = np.array(observed)
    ra = np.arctan2(observed[:, 1], observed[:, 0])
    east = np.column_stack((-np.sin(ra), np.cos(ra), np.zeros(len(ra))))
    north = np.cross(observed, east); times = offsets+60000.
    starts, epochs, costs = range_starts(times, observers, observed, east, north, 100.)
    assert len(starts) > 0
    values = (times, observed, observers, np.repeat(np.arange(3), 4))
    result = fit_chain(PreparedSeed(values), [0, 1, 2], extra_seeds=list(zip(starts, epochs)), start_policy='provided')
    assert result is not None and result['rms_arcsec'] < .05
    assert result['observations'] == 12


def test_fast_radial_light_time_matches_independent_retarded_solution():
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    from magictools.itf.derivatives import residual_jacobian
    from magictools.itf.kernels import MU, C
    # Stress the mathematical propagator; this is not a population prior or a
    # claim that such an extreme fitted state represents a plausible discovery.
    state = np.array([1000., 30., 5., 50., .1, 0.])
    offsets = np.array([-1., 0., 1.]); observers = np.tile([1., 0., 0.], (3, 1))
    def rhs(t, y): return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
    trajectory = solve_ivp(rhs, [0., -20.], state, method='DOP853', rtol=2e-12, atol=1e-13, dense_output=True).sol
    observed = []
    for reception, observer in zip(offsets, observers):
        emitted = brentq(lambda t: t+np.linalg.norm(trajectory(t)[:3]-observer)/C-reception,
                         -15., reception, xtol=1e-13)
        delta = trajectory(emitted)[:3]-observer; observed.append(delta/np.linalg.norm(delta))
    observed = np.array(observed); ra = np.arctan2(observed[:, 1], observed[:, 0])
    east = np.column_stack((-np.sin(ra), np.cos(ra), np.zeros(3))); north = np.cross(observed, east)
    residual, jacobian = residual_jacobian(state, offsets, observers, observed, east, north, .3)
    assert np.max(np.abs(residual)) < 1e-6
    assert np.isfinite(jacobian).all()


def test_compiled_gauss_preserves_positive_roots(injected_catalogue):
    from magictools.itf.fit import Geometry, gauss_seeds
    from magictools.itf.harmonic import gauss_starts
    c, sites, _ = injected_catalogue
    times, directions, observers, _ = Geometry(c, sites).get([0, 2, 5])
    expected = sorted(gauss_seeds(times, directions, observers), key=lambda x: np.linalg.norm(x[0][:3]))
    actual = sorted(gauss_starts(times, directions, observers), key=lambda x: np.linalg.norm(x[0][:3]))
    assert len(actual) == len(expected)
    for (a, ta), (e, te) in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-7, atol=1e-9)
        assert ta == te


def test_cascade_checkpoint_preserves_deferred_queue_and_retry_diagnostics(tmp_path):
    import json, zlib
    from magictools.itf.campaign_fit import open_checkpoint, commit_batch, policy_counts
    db = open_checkpoint(tmp_path/'checkpoint.sqlite', {'policy': 'cascade'})
    pack = lambda rows: zlib.compress(json.dumps(rows).encode())
    commit_batch(db, (0, 0, 3, bytes([1, 2, 0]), [(1, pack({'modes': []}))], [], 1.,
        pack([{'seed': 0, 'disposition': 'deferred_full_grid'}, {'seed': 1, 'disposition': 'accepted_fit'}])))
    commit_batch(db, (0, 0, 3, bytes([0, 0, 1]), [], [], 1.,
        pack([{'seed': 2, 'disposition': 'deferred_full_grid'}])))
    counts = policy_counts(db, 3, 'cascade')
    assert counts['seeds_completed'] == 3
    assert counts['seeds_deferred_full_grid'] == 2
    assert counts['seeds_pending_deeper_initializations'] == 3
    records = json.loads(zlib.decompress(db.execute('SELECT body FROM diagnostics').fetchone()[0]))
    assert {r['seed'] for r in records} == {0, 1, 2}
    db.close()


def test_jpl_atlas_conversion_preserves_exact_ground_station_geometry(injected_catalogue,tmp_path,monkeypatch):
    from magictools.itf.campaign_geometry import prepare_atlas
    from magictools.itf.campaign_geometry_fast import prepare_observation_ids
    from magictools.itf.campaign_jpl_geometry import prepare_jpl_atlas,JPLAtlasGeometry
    from magictools.itf.nbody import JPLGeometry
    import magictools.itf.campaign_jpl_geometry as module
    import erfa
    from astropy.time import Time
    class SyntheticEarth:
        def __init__(self,path):self.kernel=self
        def close(self):pass
        def earth(self,t):
            time=Time(t,format='mjd',scale='tdb');p,_=erfa.epv00(time.jd1,time.jd2)
            return p['p']+np.array([1e-5,-2e-5,3e-5])
    monkeypatch.setattr(module,'ExactEarth',SyntheticEarth)
    c,sites,sitefile=injected_catalogue;campaign=tmp_path/'campaign';campaign.mkdir()
    epfile=tmp_path/'ephemeris';epfile.write_bytes(b'controlled test ephemeris')
    prepare_atlas(c.path,sitefile,campaign/'fit-geometry',lambda _:None)
    prepare_observation_ids(c.path,campaign/'observation-identities')
    prepare_jpl_atlas(c.path,sitefile,campaign,epfile,lambda _:None)
    ephemeris=SyntheticEarth(epfile);reference=JPLGeometry(c,sites,ephemeris)
    actual=JPLAtlasGeometry(c.path,sitefile,campaign,ephemeris,epfile)
    indices=[0,2,5];reference.prepare(indices)
    for i,(a,e) in enumerate(zip(actual.get(indices),reference.get(indices))):
        if i==2:np.testing.assert_allclose(a,e,rtol=0,atol=5e-16)
        else:np.testing.assert_array_equal(a,e)
    # Idempotent completion and exact ephemeris provenance are enforced.
    prepare_jpl_atlas(c.path,sitefile,campaign,epfile,lambda _:None)
    epfile.write_bytes(b'other ephemeris')
    with pytest.raises(ValueError,match='different ephemeris'):
        JPLAtlasGeometry(c.path,sitefile,campaign,ephemeris,epfile)
