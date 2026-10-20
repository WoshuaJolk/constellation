from itertools import combinations
import json

import numpy as np
import pytest

from magictools.ejector import run_ejector, scan_convergences
from magictools.ejector_continuous import ball_interval, scan_segments, orbit_segments
from magictools.galactic import GalacticPotential


def test_catches_event_entirely_between_grid_points_and_tangency():
    start = np.zeros((16, 3, 6))
    start[:, :, :3] = [[-10, 0, 0], [0, -10, 0], [10, 10, 0]]
    end = -start
    old, _ = scan_convergences(np.stack([start, end]), [0, 1], .1)
    rows, _ = scan_segments([(start, end)], [0, 1], .1, 16)
    assert old == []
    assert len(rows) == 1
    assert rows[0]['lookback_myr'] == pytest.approx(.5)
    assert rows[0]['support'] == 1
    np.testing.assert_allclose(rows[0]['origin_given_convergence_percentiles_pc'], 0)
    lo, hi = ball_interval(np.array([[-1., 1, 0]]), np.array([[2., 0, 0]]), 1)
    assert lo[0] == hi[0] == .5


def test_sparse_gate_matches_independent_dense_time_bruteforce():
    rng = np.random.default_rng(193)
    start = rng.normal(size=(23, 9, 6))
    end = rng.normal(size=(23, 9, 6))
    radius, threshold = 1.1, .2
    rows, _ = scan_segments([(start, end)], [0, 1], radius, len(start), min_support=threshold)
    found = {r['indices']: r for r in rows}
    grid = np.linspace(0, 1, 4001)
    for group in combinations(range(9), 3):
        a, b = start[:, group, :3], end[:, group, :3]
        points = a[None] + grid[:, None, None, None] * (b - a)[None]
        distances = np.linalg.norm(points - points.mean(axis=2, keepdims=True), axis=-1)
        support = np.mean(distances.max(axis=2) <= radius, axis=1)
        if support.max() >= threshold:
            assert group in found
            assert found[group]['support'] >= support.max() - 1e-12
        if group in found:
            row = found[group]
            at = a + row['lookback_myr'] * (b - a)
            d = np.linalg.norm(at - at.mean(axis=1, keepdims=True), axis=-1)
            # Floating-point endpoints may differ by ulps on an inclusive boundary.
            assert np.mean(d.max(axis=1) <= radius + 1e-12) >= row['support']
            assert row['support'] - support.max() <= 1 / len(start) + 1e-12


def test_pair_encounters_at_different_times_do_not_become_common_origin():
    start = np.zeros((16, 3, 6))
    end = start.copy()
    start[:, :, 0] = [-2, 0, 3]
    end[:, :, 0] = [2, 0, -1]
    rows, stats = scan_segments([(start, end)], [0, 1], .1, 16)
    assert stats['group_segment_evaluations'] == 1
    assert rows == []


def test_simultaneous_support_not_union_over_draws():
    start = np.zeros((16, 3, 6)); end = start.copy()
    for k in range(16):
        t = .1 + .8*k/15
        v = np.array([[-10, 0, 0], [10, 0, 0], [0, 10, 0]])
        start[k, :, :3] = -t * v
        end[k, :, :3] = (1-t) * v
    rows, _ = scan_segments([(start, end)], [0, 1], .01, 16, min_support=.2)
    assert rows == []


def test_stationary_and_multiple_disjoint_encounters():
    near = np.zeros((16, 3, 6)); far = near.copy()
    far[:, :, 0] = [-10, 0, 10]
    rows, _ = scan_segments([(near, far), (far, near)], [0, 1, 2], 1, 16)
    np.testing.assert_allclose(rows[0]['convergence_intervals_myr'], [[0, .1], [1.9, 2]])
    rows, _ = scan_segments([(near, near)], [0, 1], 1, 16)
    assert rows[0]['convergence_intervals_myr'] == [[0, 1]]


def test_caps_fail_closed():
    states = np.zeros((16, 8, 6))
    with pytest.raises(ValueError, match='Pair-check cap'):
        scan_segments([(states, states)], [0, 1], 1, 16, max_pair_checks=1)
    with pytest.raises(ValueError, match='Group-evaluation cap'):
        scan_segments([(states, states)], [0, 1], 1, 16, max_group_evaluations=1)


def test_streaming_removes_128_star_limit_and_writes_all_tracebacks(tmp_path):
    states = np.zeros((16, 150, 6))
    states[:, :, 0] = np.arange(150) * 10000
    states[:, :3, :3] = 0
    np.savez(tmp_path/'catalogue.npz', states_pc_kms=states, source_ids=np.array([str(i) for i in range(150)]))
    value = dict(schema_version=2, epoch_label='test', catalogue='catalogue.npz', potential=dict(components=[]),
                 tracebacks_output='paths', search=dict(lookback_max_myr=1, time_step_myr=.5, radius_pc=1, n_controls=2))
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(value))
    report = run_ejector(path)
    assert report['search_summary']['catalogue_size'] == 150
    assert report['candidates'][0]['star_ids'] == ['0', '1', '2']
    assert report['candidates'][0]['search_wide_velocity_permutation_p'] == 1
    assert np.load(tmp_path/'paths/states.npy', mmap_mode='r').shape == (3, 16, 150, 6)
    assert (tmp_path/'paths/complete.json').exists()
    with pytest.raises(FileExistsError):
        run_ejector(path)


def test_stream_numerics_matches_independent_dop853():
    from scipy.integrate import solve_ivp
    from magictools.galactic import KMS_TO_PC_MYR
    potential = GalacticPotential(dict(components=[dict(type='plummer', mass_msun=1e11, scale_pc=1000)]))
    states = np.array([[[8000., 0, 100, 0, 230, 10]]])
    times = np.linspace(0, 5, 11)
    diagnostics = {}
    pairs = list(orbit_segments(states, times, potential, .01, diagnostics))
    def rhs(t, y):
        return np.r_[y[3:] * KMS_TO_PC_MYR, potential.acceleration(y[:3]) / KMS_TO_PC_MYR]
    truth = solve_ivp(rhs, [0, -5], states.ravel(), rtol=1e-12, atol=1e-12, method='DOP853').y[:, -1]
    assert np.linalg.norm(pairs[-1][1].ravel()[:3] - truth[:3]) < 1e-4
    assert diagnostics['max_midpoint_chord_error_pc'] > 0


def test_counterpart_comparison_at_between_grid_event(tmp_path):
    from magictools.galactic import KMS_TO_PC_MYR
    states = np.zeros((16, 3, 6))
    velocities = np.array([[100,0,0],[0,100,0],[-100,-100,0]])
    states[:,:,:3] = .345 * velocities * KMS_TO_PC_MYR
    states[:,:,3:] = velocities
    np.savez(tmp_path/'stars.npz', states_pc_kms=states, source_ids=['a','b','c'])
    np.savez(tmp_path/'counterparts.npz',states_pc_kms=np.zeros((16,1,6)),source_ids=['cluster'])
    manifest=dict(schema_version=2,epoch_label='test',catalogue='stars.npz',counterparts='counterparts.npz',
                  potential=dict(components=[]),search=dict(lookback_max_myr=1,time_step_myr=.5,radius_pc=1),tracebacks_output='paths')
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    result=run_ejector(path)
    candidate=result['candidates'][0]
    assert candidate['lookback_myr'] == pytest.approx(.345)
    assert candidate['supplied_counterpart_comparisons'][0]['conditional_proximity_fraction'] == 1
    assert candidate['relative_rms_speed_given_convergence_percentiles_kms'][1] > 100


def test_missing_rv_nonidentifiability_counterexample():
    from research.ejector_catalogue.identifiability import ambiguity
    report=ambiguity()
    assert report['max_transverse_velocity_difference_kms'] < 1e-12
    first,second=report['histories']
    assert first['origin_pc'] != second['origin_pc']
    assert first['radial_velocity_kms'] != second['radial_velocity_kms']
    assert max(row['convergence_error_pc'] for row in report['histories']) < 1e-10


def test_radial_preserving_control_keeps_selection_geometry():
    from magictools.ejector_continuous import rotate_tangential_velocities
    rng=np.random.default_rng(78)
    observer=np.array([-8122.,0,20.8,12.9,245.6,7.78])
    states=rng.normal(size=(16,20,6))*[100,100,100,50,50,50]+observer
    angles=rng.uniform(0,2*np.pi,20)
    changed=rotate_tangential_velocities(states,observer,angles)
    direction=states[...,:3]-observer[:3]
    direction/=np.linalg.norm(direction,axis=-1)[...,None]
    first,second=states[...,3:]-observer[3:],changed[...,3:]-observer[3:]
    np.testing.assert_allclose(np.sum(first*direction,axis=-1),np.sum(second*direction,axis=-1),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(first,axis=-1),np.linalg.norm(second,axis=-1),atol=1e-12)
    np.testing.assert_array_equal(states[...,:3],changed[...,:3])
    assert not np.allclose(first,second)


def test_rotation_groups_keep_a_pair_together(tmp_path, monkeypatch):
    import magictools.ejector_continuous as module
    states=np.zeros((16,3,6));states[:,:,0]=[100,100,200];states[:,:,4]=[20,20,30]
    np.savez(tmp_path/'catalogue.npz',states_pc_kms=states,source_ids=['a','b','c'],rotation_groups=['pair','pair','single'])
    spec=dict(schema_version=2,epoch_label='test',catalogue='catalogue.npz',potential=dict(components=[]),
              tracebacks_output='paths',search=dict(lookback_max_myr=1,time_step_myr=.5,radius_pc=1000,
              n_controls=2,control_method='tangent_rotation',observer_state_pc_kms=[0,0,0,0,0,0]))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(spec))
    original=module.rotate_tangential_velocities
    seen=[]
    def capture(states,observer,angles):
        out=original(states,observer,angles)
        seen.append(angles.copy())
        np.testing.assert_allclose(out[:,0,3:],out[:,1,3:])
        return out
    monkeypatch.setattr(module,'rotate_tangential_velocities',capture)
    result=run_ejector(path)
    assert result['controls']['rotation_group_count']==2
    assert len(seen)==3 # initial validation and two actual controls
    assert seen[1][0]==seen[1][1] and seen[1][0]!=seen[1][2]
    assert result['candidates'][0]['search_wide_velocity_permutation_p'] is None


def test_tighter_radius_rescan_matches_full_catalogue(tmp_path):
    from research.ejector_catalogue.tighten_radius import tighten
    states=np.zeros((16,150,6));states[:,:,0]=np.arange(150)*10000
    states[:,:3,:3]=0;states[:,1,0]=.1
    np.savez(tmp_path/'catalogue.npz',states_pc_kms=states,source_ids=np.array([str(i) for i in range(150)]))
    spec=dict(schema_version=2,epoch_label='test',catalogue='catalogue.npz',potential=dict(components=[]),
              tracebacks_output='paths',search=dict(lookback_max_myr=1,time_step_myr=.5,radius_pc=10))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(spec));run_ejector(path)
    for radius in [1.,.01]:
        result=tighten(tmp_path/'paths',radius,tmp_path/f'tight-{radius}.json')
        full,_=scan_segments([(states,states),(states,states)],[0,.5,1],radius,16)
        assert len(result['candidates'])==len(full)
        assert result['parent_catalogue_size']==150 and result['searched_member_union']==3
    with pytest.raises(ValueError,match='positive radius'):
        tighten(tmp_path/'paths',11,tmp_path/'bad.json')


def test_tiny_positive_support_does_not_include_zero_support_gaps():
    start=np.zeros((2,3,6));end=start.copy()
    velocity=np.array([[-10.,0,0],[10.,0,0],[0,10.,0]])
    for k,t in enumerate([.2,.8]):
        start[k,:,:3]=-t*velocity;end[k,:,:3]=(1-t)*velocity
    rows,_=scan_segments([(start,end)],[0,1],.01,2,min_support=1e-20)
    assert len(rows)==1
    assert len(rows[0]['convergence_intervals_myr'])==2
    assert all(not lo <= .5 <= hi for lo,hi in rows[0]['convergence_intervals_myr'])
