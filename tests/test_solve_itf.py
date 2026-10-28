"""Independent dynamics checks and whole-pipeline multi-night linkage controls."""
import json
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

pytest.importorskip('numba')
from magictools.itf.catalogue import TRACK_DTYPE, build, restore, unit_vectors
from magictools.itf.cli import main
from magictools.itf.fit import Geometry, fit_chain, read_sites
from magictools.itf.kernels import ARCSEC, C, MU, directions, parse_numeric, position
from magictools.itf.search import SearchConfig, candidate_edges, solve


def mpc_line(identifier, mjd, ra, dec, site='500'):
    from astropy.time import Time
    date = Time(mjd, format='mjd', scale='utc').ymdhms
    day = date.day+(date.hour*3600+date.minute*60+date.second)/86400
    ra_hours = ra/15
    h = int(ra_hours)
    minute = int((ra_hours-h)*60)
    sec = ((ra_hours-h)*60-minute)*60
    degree = int(abs(dec))
    dm = int((abs(dec)-degree)*60)
    ds = ((abs(dec)-degree)*60-dm)*60
    return (f'{identifier:>12}  C{date.year:04d} {date.month:02d} {day:09.6f}'
            f'{h:02d} {minute:02d} {sec:06.3f}{"-" if dec<0 else "+"}'
            f'{degree:02d} {dm:02d} {ds:05.2f}'+ ' '*21+site+'\n')


def test_parser_roundtrip_invalid_date_and_unsupported_observer():
    line = mpc_line('ABC', 60000.123456, 359.12345, -23.45678)
    assert len(line.rstrip('\n')) == 80
    satellite = line[:14]+'S'+line[15:]
    invalid = line[:20]+'02 30.123456'+line[32:]
    raw = np.frombuffer(''.join(s.rstrip('\n') for s in (line, satellite, invalid)).encode(), np.uint8).reshape(-1, 80)
    values, status = parse_numeric(raw)
    assert status.tolist() == [0, 2, 1]
    np.testing.assert_allclose(values[0], [60000.123456, 359.12345, -23.45678], atol=3e-6, rtol=0)


@pytest.mark.parametrize('speed', [.012, np.sqrt(2*MU), .035])
def test_compiled_all_conic_kernel_against_independent_ode(speed):
    state = np.array([1., 0., 0., 0., speed, 0.])
    def derivative(t, y):
        return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
    for dt in [-100., 0., 100.]:
        expected = solve_ivp(derivative, [0, dt or 1e-15], state, method='DOP853',
                             rtol=2e-12, atol=1e-14).y[:3, -1]
        np.testing.assert_allclose(position(state, dt), expected, atol=2e-10, rtol=0)
    observers = np.tile([.8, -.1, .03], (3, 1))
    offsets = np.array([-10., 0., 10.])
    actual = directions(state, offsets, observers)
    for k, dt in enumerate(offsets):
        emitted = dt
        for _ in range(8):
            expected = solve_ivp(derivative, [0, emitted or 1e-15], state, method='DOP853',
                                 rtol=2e-12, atol=1e-14).y[:3, -1]-observers[k]
            emitted = dt-np.linalg.norm(expected)/C
        np.testing.assert_allclose(actual[k], expected/np.linalg.norm(expected), atol=1e-10)


def test_cache_wrap_split_dedup_and_identity(tmp_path):
    source = tmp_path/'input.txt'
    rows = [mpc_line('A', 60000., 359.999, 20), mpc_line('A', 60000.02, .001, 20),
            mpc_line('A', 60002., .1, 20), mpc_line('A', 60002.02, .102, 20)]
    source.write_text(''.join(rows+[rows[0]]))
    cat = build(source, tmp_path/'cache')
    assert len(cat.tracks) == 2
    assert len(cat.obs) == 4
    assert cat.manifest['counts']['duplicate_rows_within_key'] == 1
    assert np.max(cat.tracks['rms']) < .001
    assert np.linalg.norm(cat.tracks[0]['v']) < .01
    assert build(source, tmp_path/'cache').manifest == cat.manifest
    unpacked = restore(cat.path/'catalogue.npz', cat.path/'manifest.json', tmp_path/'restored')
    np.testing.assert_allclose(unpacked.obs['t'], cat.obs['t'], atol=1e-11, rtol=0)
    np.testing.assert_array_equal(unpacked.obs['ra_mas'], cat.obs['ra_mas'])
    np.testing.assert_array_equal(unpacked.obs['dec_mas'], cat.obs['dec_mas'])
    np.testing.assert_allclose(unpacked.tracks['v'], cat.tracks['v'], atol=1e-10, rtol=0)
    broken = tmp_path/'broken.npz'
    broken.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore(broken, cat.path/'manifest.json', tmp_path/'bad-restore')
    source.write_text(''.join(rows[:2]))
    with pytest.raises(ValueError, match='different input'):
        build(source, tmp_path/'cache')


@pytest.fixture
def injected_catalogue(tmp_path):
    """Six tracklets from independent DOP853 dynamics, with weak measurement noise."""
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    times = 60000+np.repeat([0., 1., 3., 5., 7., 9.], 4)+np.tile([0., .015, .03, .045], 6)
    utc = Time(times, format='mjd', scale='utc')
    t = utc.tdb.mjd
    earth, _ = get_body_barycentric_posvel('earth', utc.tdb)
    sun, _ = get_body_barycentric_posvel('sun', utc.tdb)
    observers = (earth.xyz-sun.xyz).to_value(u.au).T
    state = np.array([2.4, .7, .3, -.004, .009, .002])
    epoch = t.mean()
    def derivative(t, y):
        return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
    forward = solve_ivp(derivative, [0., 12.], state, method='DOP853', rtol=2e-12, atol=1e-14, dense_output=True).sol
    backward = solve_ivp(derivative, [0., -12.], state, method='DOP853', rtol=2e-12, atol=1e-14, dense_output=True).sol
    rng = np.random.default_rng(17)
    lines = []
    for k, reception in enumerate(t):
        emitted = reception-epoch
        for _ in range(8):
            xyz = (forward if emitted >= 0 else backward)(emitted)[:3]-observers[k]
            emitted = reception-epoch-np.linalg.norm(xyz)/C
        ra = np.degrees(np.arctan2(xyz[1], xyz[0])) % 360
        dec = np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1])))
        ra += rng.normal(0, .02)/3600/np.cos(np.deg2rad(dec))
        dec += rng.normal(0, .02)/3600
        lines.append(mpc_line(f'INJ{k//4}', times[k], ra, dec))
    source = tmp_path/'injection.txt'
    source.write_text(''.join(lines))
    sitefile = tmp_path/'ObsCodes.html'
    sitefile.write_text('500   0.0000 0.00000 +0.00000 Geocenter\n')
    return build(source, tmp_path/'cache'), read_sites(sitefile), sitefile


def test_real_orbit_fit_and_unrelated_chain_rejected(injected_catalogue):
    cat, sites, _ = injected_catalogue
    geo = Geometry(cat, sites)
    result = fit_chain(geo, (0, 2, 5), deadline=time.monotonic()+25)
    assert result is not None
    assert result['rms_arcsec'] < .1
    assert result['modes'][0]['jacobian_rank'] == 6
    assert result['probability_bic'] > .9
    # Keep dates/observer geometry but corrupt a whole tracklet, not one outlier.
    original = geo.track(2)
    corrupt = original[2].copy()
    corrupt[:, [0, 1]] = corrupt[:, [1, 0]]
    original_track = geo.track
    geo.track = lambda i: (original[0], original[1], corrupt, original[3]) if i == 2 else original_track(i)
    assert fit_chain(geo, (0, 2, 5), deadline=time.monotonic()+25) is None


def test_full_search_grows_three_to_six_and_reports_limits(injected_catalogue):
    cat, sites, _ = injected_catalogue
    events=[]
    report = solve(cat, sites, SearchConfig(seconds=45, max_seeds=30),audit=events.append)
    assert len([e for e in events if e['event']=='seed_bank'])==1
    submitted=[tuple(e['indices']) for e in events if e['event']=='submitted']
    completed=[tuple(e['indices']) for e in events if e['event']=='fit']
    assert sorted(submitted)==sorted(completed)
    assert report['counts']['full_catalogue_scan_complete']
    assert report['counts']['tracklets_scanned'] == 6
    assert any(len(row['tracklets']) == 6 for row in report['candidates'])
    assert {3, 6} <= {len(row['tracklets']) for row in report['candidates']}
    assert report['counts']['individual_extension_subsets_covered'] > 0
    assert report['probabilities_calibrated'] is False
    assert report['exhaustive'] is False
    assert report['candidates'][0]['rms_arcsec'] < .1
    json.dumps(report, allow_nan=False)
    short = solve(cat, sites, SearchConfig(seconds=.000001))
    assert short['status'] == 'time_budget_exhausted'
    assert not short['counts']['full_catalogue_scan_complete']


def test_two_tracklet_training_start_is_opt_in_and_uses_only_training_data(injected_catalogue):
    cat,sites,_=injected_catalogue;geo=Geometry(cat,sites)
    original=geo.get
    def four_points(indices):
        t,u,observer,labels=original(indices)
        keep=np.concatenate([np.flatnonzero(labels==i)[[0,-1]] for i in indices])
        return t[keep],u[keep],observer[keep],labels[keep]
    geo.get=four_points
    assert fit_chain(geo,[0,3]) is None
    row=fit_chain(geo,[0,3],allow_two_tracklets=True,broad_starts=True)
    assert row is not None and row['observations']==4
    assert row['rms_arcsec']<.1
    assert all(np.isfinite(m['state_au_au_per_day']).all() for m in row['modes'])


def test_cli_rejects_invalid_budget_without_creating_output(tmp_path):
    output = tmp_path/'result'
    with pytest.raises(SystemExit) as exc:
        main(['run', '--cache', str(tmp_path/'none'), '--observatories', str(tmp_path/'none'),
              '--output', str(output), '--seconds', '-1'])
    assert exc.value.code == 2
    assert not output.exists()


def test_candidate_scan_crosses_ra_zero_near_pole_without_losing_tracklets():
    tracks = np.zeros(4, dtype=TRACK_DTYPE)
    tracks['id'] = [b'A', b'B', b'C', b'SINGLE']
    tracks['site'] = b'500'
    tracks['count'] = [3, 3, 3, 1]
    tracks['span'] = [.04, .04, .04, 0.]
    tracks['t'] = [60000.1, 60001.1, 60002.1, 60003.1]
    ra = np.array([359.9, 0., .1, 200.])
    dec = np.array([89.9, 89.9, 89.9, 0.])
    tracks['u'] = unit_vectors(ra, dec)
    tracks['v'] = (unit_vectors(ra+.001, dec)-unit_vectors(ra-.001, dec))/.02
    counts = {}
    edges = candidate_edges(tracks, SearchConfig(), time.monotonic()+5, lambda s: None, counts)
    assert {tuple(map(int, row[:2])) for row in edges} == {(0, 1), (0, 2), (1, 2)}
    assert counts['tracklets_total'] == 4
    assert counts['tracklets_scanned'] == 3
    assert counts['rate_unavailable_tracklets'] == 1
    assert counts['full_catalogue_scan_complete']


@pytest.mark.parametrize('speed', [.012, np.sqrt(2*MU), .035])
def test_analytic_state_jacobian_against_independent_variational_ode(speed):
    from magictools.itf.derivatives import state_geometry
    state = np.array([1., 0., 0., 0., speed, .001])
    def variational(t, y):
        r, v = y[:3], y[3:6]
        radius = np.linalg.norm(r)
        a = np.zeros((6, 6))
        a[:3, 3:] = np.eye(3)
        a[3:, :3] = -MU*np.eye(3)/radius**3+3*MU*np.outer(r, r)/radius**5
        return np.r_[v, -MU*r/radius**3, (a@y[6:].reshape(6, 6)).ravel()]
    for dt in (-70., 70.):
        expected = solve_ivp(variational, [0, dt], np.r_[state, np.eye(6).ravel()],
                             method='DOP853', rtol=2e-12, atol=1e-14).y[:, -1]
        p, v, jac = state_geometry(state, dt)
        np.testing.assert_allclose(p, expected[:3], atol=2e-10, rtol=0)
        np.testing.assert_allclose(v, expected[3:6], atol=2e-11, rtol=0)
        np.testing.assert_allclose(jac, expected[6:].reshape(6, 6)[:3], atol=2e-8, rtol=1e-9)


def test_analytic_light_time_residual_jacobian():
    from magictools.itf.derivatives import residual_jacobian
    state = np.array([2.4, .7, .3, -.004, .009, .002])
    offsets = np.array([-12., -4., 0., 6., 13.])
    observers = np.column_stack([np.cos(offsets*.017), np.sin(offsets*.017), np.zeros(len(offsets))])
    u = directions(state, offsets, observers)
    ra = np.arctan2(u[:, 1], u[:, 0])
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros(len(ra))])
    north = np.cross(u, east)
    residual, jac = residual_jacobian(state, offsets, observers, u, east, north, .3)
    assert np.max(abs(residual)) < 1e-7
    finite = np.empty_like(jac)
    for k in range(6):
        step = 1e-6 if k < 3 else 1e-8
        delta = np.eye(6)[k]*step
        plus = residual_jacobian(state+delta, offsets, observers, u, east, north, .3)[0]
        minus = residual_jacobian(state-delta, offsets, observers, u, east, north, .3)[0]
        finite[:, k] = (plus-minus)/(2*step)
    np.testing.assert_allclose(jac, finite, atol=.015, rtol=1e-6)


def test_parallel_scan_and_batched_geometry_preserve_inputs(injected_catalogue):
    cat, sites, _ = injected_catalogue
    serial_counts, parallel_counts = {}, {}
    serial = candidate_edges(cat.tracks, SearchConfig(workers=1), time.monotonic()+10,
                             lambda message: None, serial_counts)
    parallel = candidate_edges(cat.tracks, SearchConfig(workers=2), time.monotonic()+10,
                               lambda message: None, parallel_counts)
    order = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]
    np.testing.assert_allclose(order(serial), order(parallel), atol=0, rtol=0)
    for key in ('edges_proposed', 'tracklets_scanned', 'day_pairs_scanned', 'neighbor_queries_at_cap'):
        assert serial_counts[key] == parallel_counts[key]
    reference, batched = Geometry(cat, sites), Geometry(cat, sites)
    batched.prepare(range(len(cat.tracks)))
    for index in range(len(cat.tracks)):
        for expected, actual in zip(reference.track(index)[1:], batched.track(index)[1:]):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_compiled_optimizer_matches_reference_on_independent_injection(injected_catalogue):
    cat, sites, _ = injected_catalogue
    geo = Geometry(cat, sites)
    geo.prepare([0, 2, 5])
    compiled = fit_chain(geo, (0, 2, 5), jacobian_backend='compiled')
    reference = fit_chain(geo, (0, 2, 5), jacobian_backend='finite_difference')
    assert compiled is not None and reference is not None
    assert abs(compiled['rms_arcsec']-reference['rms_arcsec']) < 1e-5
    assert abs(compiled['inclination_deg']-reference['inclination_deg']) < .01


def test_same_night_attachment_and_maximal_view(injected_catalogue, tmp_path):
    from magictools.itf.cli import format_rows
    cat, sites, _ = injected_catalogue
    lines = (cat.path.parent/'injection.txt').read_text().splitlines(keepends=True)
    # Split a night into two independently designated, two-observation arcs.
    # No seed can include both: the three separated-epoch seed rule still holds.
    lines[:2] = [f'{"SPLIT":>12}'+line[12:] for line in lines[:2]]
    source = tmp_path/'split.txt'
    source.write_text(''.join(lines))
    split = build(source, tmp_path/'split-cache')
    report = solve(split, sites, SearchConfig(seconds=45, max_seeds=40))
    full = [row for row in report['candidates'] if len(row['tracklets']) == 7]
    assert len(full) == 1
    assert full[0]['observations'] == 24
    assert full[0]['rms_arcsec'] < .1
    assert full[0]['maximal'] and full[0]['contains_ranks']
    assert report['counts']['accepted_chains_with_close_epochs'] > 0
    assert report['counts']['maximal_candidates'] == 1
    assert report['counts']['contained_candidates'] == len(report['candidates'])-1
    text = format_rows(report, maximal_only=True)
    assert text.count('BIC score ') == 1
    assert 'SPLIT' in text
    json.dumps(report, allow_nan=False)


def test_containment_keeps_partial_overlap_alternatives():
    from magictools.itf.search import annotate_candidates
    tracks = np.zeros(8, TRACK_DTYPE)
    tracks['id'] = [str(i).encode() for i in range(8)]
    tracks['site'] = b'500'
    tracks['t'] = np.arange(8)
    rows = [dict(tracklet_indices=list(indices), modes=[dict(local_uncertainty=dict(local_linearization_weak=False))])
            for indices in [(0, 1, 2, 3), (0, 1, 2), (2, 3, 4), (5, 6, 7)]]
    counts = annotate_candidates(rows, tracks)
    assert counts['maximal_candidates'] == 3
    assert rows[0]['contains_ranks'] == [2]
    assert rows[1]['contained_by_ranks'] == [1]
    assert rows[0]['partially_overlapping_ranks'] == [3]
    assert rows[2]['partially_overlapping_ranks'] == [1, 2]
    assert rows[3]['overlapping_ranks'] == []


def test_local_errors_against_independent_covariance_and_ode():
    from magictools.itf.fit import local_uncertainty, orbital_elements
    state = np.array([2.4, .7, .3, -.004, .009, .002])
    observer = np.array([1., .1, 0.])
    jac = np.random.default_rng(42).normal(size=(20, 6))*np.array([2e5]*3+[7e6]*3)
    result = local_uncertainty(state, jac, 3., observer)
    def values(x):
        def derivative(t, y):
            return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
        p = solve_ivp(derivative, [0, 3], x, method='DOP853', rtol=2e-12, atol=1e-14).y[:3, -1]
        return np.array([np.linalg.norm(p-observer), orbital_elements(x)['inclination_deg'],
                         np.dot(x[3:], x[3:])/2-MU/np.linalg.norm(x[:3])])
    gradients = np.empty((3, 6))
    for k in range(6):
        step = 1e-6 if k < 3 else 1e-8
        change = np.eye(6)[k]*step
        gradients[:, k] = (values(state+change)-values(state-change))/(2*step)
    covariance = np.linalg.inv(jac.T@jac)
    expected = np.sqrt(np.diag(gradients@covariance@gradients.T))
    names = ['range_sigma_au', 'inclination_sigma_deg', 'energy_sigma_au2_per_day2']
    np.testing.assert_allclose([result[name] for name in names], expected, rtol=1e-7)
    doubled = local_uncertainty(state, jac/2, 3., observer)
    np.testing.assert_allclose([doubled[name] for name in names], 2*expected, rtol=1e-7)
    jac[:, 5] = jac[:, 4]
    singular = local_uncertainty(state, jac, 3., observer)
    assert singular['scaled_jacobian_rank'] == 5
    assert singular['local_linearization_weak']
    assert all(singular[name] is None for name in names)


def test_cli_fresh_cache_compiles_in_background_and_writes_both_views(injected_catalogue, tmp_path):
    import os
    import subprocess
    import sys
    cat, _, sites = injected_catalogue
    output = tmp_path/'cli-output'
    completed = subprocess.run([
        sys.executable, '-m', 'magictools.itf.cli', 'run',
        '--input', str(cat.path.parent/'injection.txt'), '--cache', str(tmp_path/'cli-cache'),
        '--observatories', str(sites), '--output', str(output),
        '--workers', '2', '--max-seeds', '30', '--seconds', '60', '--top', '2'],
        text=True, capture_output=True, timeout=90, env=dict(os.environ, OPENBLAS_NUM_THREADS='1'))
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'bounded_search_complete'
    assert report['compiler_warmup']['completed']
    assert report['compiler_warmup']['exit_code'] == 0
    assert (output/'maximal_candidates.txt').read_text().count('BIC score ') == 1
    assert (output/'candidates.txt').read_text().count('BIC score ') > 1
    assert completed.stdout.count('BIC score ') == 1


def test_spawn_fit_worker_reopens_catalogue_and_prepares_new_neighbors(injected_catalogue):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from magictools.itf.search import _initialize_fit_worker, _fit_worker
    cat, sites, _ = injected_catalogue
    deadline = time.monotonic()+60
    reference = fit_chain(Geometry(cat, sites), (0, 2, 5), deadline=deadline)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_fit_worker,
                             initargs=(None, str(cat.path), sites, np.array([0, 2]), deadline)) as pool:
        result = pool.submit(_fit_worker, (0, 2, 5), dict(deadline=deadline)).result()
    assert result is not None and reference is not None
    assert result['tracklet_indices'] == reference['tracklet_indices']
    assert abs(result['rms_arcsec']-reference['rms_arcsec']) < 1e-7
    assert abs(result['probability_bic']-reference['probability_bic']) < 1e-7


@pytest.mark.parametrize('tracklets', [4, 5])
def test_joint_growth_recovers_four_and_five_tracklet_arcs(injected_catalogue, tmp_path, tracklets):
    cat, sites, _ = injected_catalogue
    lines = (cat.path.parent/'injection.txt').read_text().splitlines(keepends=True)
    source = tmp_path/'shorter.txt'
    source.write_text(''.join(lines[:4*tracklets]))
    shorter = build(source, tmp_path/'shorter-cache')
    report = solve(shorter, sites, SearchConfig(seconds=45, max_seeds=30))
    assert report['status'] == 'bounded_search_complete'
    assert max(len(row['tracklets']) for row in report['candidates']) == tracklets


def test_failed_joint_fit_falls_back_and_recovers_good_subset(injected_catalogue, tmp_path):
    cat, sites, _ = injected_catalogue
    lines = []
    for index, track in enumerate(cat.tracks):
        for observation in cat.obs[track['start']:track['start']+track['count']]:
            ra, dec = observation['ra_mas']/3600000, observation['dec_mas']/3600000
            if index == 2:
                # Inside the 6-arcsecond proposal gate, outside joint-fit tolerance.
                ra += 3/3600/np.cos(np.deg2rad(dec))
            lines.append(mpc_line(track['id'].decode().strip(), observation['t'], ra, dec))
    source = tmp_path/'contaminated.txt'
    source.write_text(''.join(lines))
    contaminated = build(source, tmp_path/'contaminated-cache')
    assert fit_chain(Geometry(contaminated, sites), range(6)) is None
    report = solve(contaminated, sites, SearchConfig(seconds=45, max_seeds=30))
    assert report['counts']['joint_extension_proposals'] > 0
    expected = {'INJ0', 'INJ1', 'INJ3', 'INJ4', 'INJ5'}
    assert any(set(row['tracklets']) == expected and row['rms_arcsec'] < .1 for row in report['candidates'])
    assert all(len(row['tracklets']) < 6 for row in report['candidates'])


@pytest.mark.parametrize('kind', ['nearby', 'retrograde', 'hyperbolic', 'distant'])
def test_broad_search_recovers_independent_orbits_excluded_by_fast_graph(tmp_path, kind):
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    from magictools.itf.fit import orbital_elements
    night = [0., 1., 2.] if kind == 'nearby' else [0., 18., 36.]
    times = 60000+np.repeat(night, 4)+np.tile([0., .001, .002, .003], 3)
    utc = Time(times, format='mjd', scale='utc')
    t = utc.tdb.mjd
    ep, ev = get_body_barycentric_posvel('earth', utc.tdb)
    sp, sv = get_body_barycentric_posvel('sun', utc.tdb)
    observers = (ep.xyz-sp.xyz).to_value(u.au).T
    velocity = (ev.xyz-sv.xyz).to_value(u.au/u.day).T
    epoch = t[4]
    states = {
        'nearby': np.r_[observers[4]+[.03, 0., 0.], velocity[4]+[0., .0001, .0001]],
        'retrograde': np.array([2.4, .7, .3, .004, -.009, -.002]),
        'hyperbolic': np.array([2.4, .7, .3, -.004, .025, .012]),
        'distant': np.array([300., 20., 40., -.0001, .0008, .0002]),
    }
    state = states[kind]
    if kind == 'retrograde':
        assert orbital_elements(state)['inclination_deg'] > 90
    if kind == 'hyperbolic':
        assert orbital_elements(state)['eccentricity'] > 1
    def derivative(_, y):
        return np.r_[y[3:], -MU*y[:3]/np.linalg.norm(y[:3])**3]
    forward = solve_ivp(derivative, [0., 50.], state, method='DOP853', rtol=2e-12,
                        atol=1e-14, dense_output=True).sol
    backward = solve_ivp(derivative, [0., -50.], state, method='DOP853', rtol=2e-12,
                         atol=1e-14, dense_output=True).sol
    lines = []
    for k, reception in enumerate(t):
        emitted = reception-epoch
        for _ in range(10):
            xyz = (forward if emitted >= 0 else backward)(emitted)[:3]-observers[k]
            emitted = reception-epoch-np.linalg.norm(xyz)/C
        ra = np.degrees(np.arctan2(xyz[1], xyz[0])) % 360
        dec = np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1])))
        lines.append(mpc_line(f'BROAD{k//4}', times[k], ra, dec))
    source = tmp_path/'control.txt'
    source.write_text(''.join(lines))
    catalogue = build(source, tmp_path/'control-cache')
    sitefile = tmp_path/'sites.txt'
    sitefile.write_text('500   0.0000 0.00000 +0.00000 Geocenter\n')
    sites = read_sites(sitefile)
    from magictools.itf.search import make_seeds
    fast = SearchConfig(workers=1)
    edges = candidate_edges(catalogue.tracks, fast, time.monotonic()+10, lambda _: None, {})
    seeds, _ = make_seeds(edges, catalogue.tracks, fast, {})
    assert not seeds, f'{kind} must exercise a previously excluded association'
    broad = SearchConfig.for_profile('broad')
    broad.workers, broad.max_seeds, broad.seconds = 1, 30, 60
    report = solve(catalogue, sites, broad)
    assert report['status'] == 'bounded_search_complete'
    assert report['counts']['curved_scan_complete']
    assert any(set(row['tracklets']) == {'BROAD0', 'BROAD1', 'BROAD2'}
               and row['rms_arcsec'] < .15 for row in report['candidates'])


def test_broad_observers_and_parallel_proposals(injected_catalogue):
    from magictools.itf.broad import approximate_observers, augment_edges
    cat, sites, _ = injected_catalogue
    # Exercise terrestrial rotation, not only the geocenter controls.
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    from astropy.time import Time
    from astropy.utils import iers
    import erfa
    sites[b'500'] = EarthLocation.from_geodetic(-155*u.deg, 20*u.deg, 4000*u.m)
    p, v, *_ = approximate_observers(cat.tracks, sites)
    t = Time(cat.tracks['t'], format='mjd', scale='utc')
    with iers.conf.set_temp('auto_download', False):
        offset, rate = sites[b'500'].get_gcrs_posvel(t)
    earth, _ = erfa.epv00(t.tdb.jd1, t.tdb.jd2)
    np.testing.assert_allclose(p, earth['p']+offset.xyz.to_value(u.au).T, atol=3e-8, rtol=0)
    np.testing.assert_allclose(v, earth['v']+rate.xyz.to_value(u.au/u.day).T, atol=2e-7, rtol=0)
    config = SearchConfig.for_profile('broad')
    config.workers = 1
    original = candidate_edges(cat.tracks, config, time.monotonic()+20, lambda _: None, {})
    serial_counts, parallel_counts = {}, {}
    serial = augment_edges(original, cat.tracks, sites, config, time.monotonic()+30, lambda _: None, serial_counts)
    config.workers = 2
    parallel = augment_edges(original, cat.tracks, sites, config, time.monotonic()+30, lambda _: None, parallel_counts)
    np.testing.assert_array_equal(serial, parallel)
    assert set(map(tuple, original[:, :2])) <= set(map(tuple, parallel[:, :2]))
    assert serial_counts['curved_edges_proposed'] == parallel_counts['curved_edges_proposed']


def test_seed_neighbors_are_independent_of_growth_branch_limit():
    from magictools.itf.search import make_seeds
    tracks = np.zeros(8, dtype=TRACK_DTYPE)
    tracks['id'] = [str(i).encode() for i in range(8)]
    tracks['t'] = np.arange(8, dtype=float)
    edges = np.array([(0, i, float(i)) for i in range(1, 8)])
    limited, _ = make_seeds(edges, tracks, SearchConfig(), {})
    broad, _ = make_seeds(edges, tracks, SearchConfig(seed_neighbors=0, branches=1), {})
    assert len(limited) == 10
    assert len(broad) == 21
    assert set(limited) < set(broad)


def test_curved_proposals_get_capacity_when_expanded_linear_graph_is_full():
    from magictools.itf.broad import merge_proposals
    protected = np.array([[0., 1., 10.], [1., 2., 20.]])
    linear = np.concatenate([protected, [[3., 4., .1], [4., 5., .2], [5., 6., .3]]])
    curved = np.array([[10., 11., 100.], [11., 12., 200.], [12., 13., 300.]])
    counts = {}
    merged = merge_proposals(protected, linear, curved, 5, counts)
    assert set(map(tuple, protected)) <= set(map(tuple, merged))
    assert (10., 11., 100.) in set(map(tuple, merged))
    assert len(merged) == 5
    assert counts['curved_edges_retained'] == 1
    assert counts['expanded_linear_edges_discarded'] == 1
    assert counts['curved_merge_edges_discarded'] == 2


def test_seed_score_pruning_matches_exhaustive_ranking_with_ties_and_aliases():
    from itertools import combinations
    from collections import defaultdict
    from magictools.itf.search import make_seeds
    rng = np.random.default_rng(731)
    tracks = np.zeros(45, dtype=TRACK_DTYPE)
    tracks['t'] = rng.integers(0, 30, len(tracks))/3
    tracks['id'] = [str(i//2).encode() for i in range(len(tracks))]
    edges = np.array([(a, b, float(rng.integers(0, 5))) for a, b in combinations(range(len(tracks)), 2)
                      if rng.uniform() < .7])
    adjacent = defaultdict(list)
    for a, b, score in edges:
        adjacent[int(a)].append((score, int(b)))
        adjacent[int(b)].append((score, int(a)))
    expected = {}
    for middle, links in adjacent.items():
        for (sa, a), (sb, b) in combinations(links, 2):
            key = tuple(sorted((a, middle, b)))
            if np.min(np.diff(np.sort(tracks['t'][list(key)]))) < .3:
                continue
            if len(set(tracks['id'][list(key)])) < 3:
                continue
            expected[key] = min(expected.get(key, np.inf), sa+sb)
    for limit in (7, 91):
        actual, _ = make_seeds(edges, tracks, SearchConfig(seed_neighbors=0, max_seeds=limit), {})
        assert actual == sorted(expected, key=lambda key: (expected[key], key))[:limit]
