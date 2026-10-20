"""Checks for sufficient local cells, complete budgets and cost accounting."""
import json
from pathlib import Path

import numpy as np
import pytest
from flint import arb, ctx

from research.kepler_transform.constructive_experiments import domain
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.local_resolution import (
    axis_error, axis_neighborhood, choose_axis, grid_plan, local_grid,
    nonnegative_inverse_bound, phase_bound,
)
from research.kepler_transform.local_hierarchy import LocalHierarchyPlan, transfer_cost
from research.kepler_transform.reception_flow import CircularObserver
from research.kepler_transform.segment_field import TensorGrid, lobatto_rule
from research.kepler_transform.source_butterfly import cardinal
from research.kepler_transform.cascade_transfer import CascadeTransfer


@pytest.fixture(autouse=True)
def precision():
    old = ctx.prec
    ctx.prec = 100
    yield
    ctx.prec = old


@pytest.fixture(scope="module")
def proof():
    path = Path(__file__).parents[1]/"research/kepler_transform/constructive_results/local.json"
    return json.loads(path.read_text())["certificates"][0]["transfer_certificate"]


def test_neumann_bound_and_noncontracting_refusal():
    E = [[arb(0) for _ in range(6)] for _ in range(6)]
    E[3][5], E[5][0], E[0][0] = arb('.4'), arb('.2'), arb('.1')
    S, check = nonnegative_inverse_bound(E)
    inverse = np.linalg.inv(np.eye(6)-np.array([[float(x) for x in row] for row in E]))
    assert check["contraction_bound"] < 1
    assert all(float(S[i][j]) >= inverse[i,j]-1e-15 for i in range(6) for j in range(6))
    assert S[3][0] > arb('.08')
    E[0][0] = arb('1.1')
    with pytest.raises(ValueError, match="contraction|weights"):
        nonnegative_inverse_bound(E)


@pytest.mark.parametrize("phase", [0., 1e-12, .2, 20., 200.])
def test_local_cells_bound_off_grid_fourier_values(phase):
    neighborhood = {"ellipse": 4., "hybrid_halfwidth": 1.}
    choice = choose_axis(neighborhood, phase, 1e-8, multiple=3)
    assert choice["cells"] % 3 == 0
    n, p = choice["cells"], choice["order"]
    x, bary = lobatto_rule(p)
    rng = np.random.default_rng(41)
    u = rng.uniform(-1, 1, 300)
    nodes = x/n + .3
    actual = cardinal(u, x, bary) @ np.exp(1j*phase*nodes)
    exact = np.exp(1j*phase*(u/n+.3))
    bound = float(axis_error(p, n, choice["ellipse"], phase, 1.).upper())
    assert np.max(abs(actual-exact)) <= bound+3e-14
    if phase == 0:
        assert p == 1


def test_high_frequency_requires_cells():
    q = choose_axis({"ellipse": 4., "hybrid_halfwidth": 1.}, 1000., 1e-10)
    assert q["cells"] > 1
    r, r0 = arb(q["ellipse"]), arb(q["neighborhood_ellipse"])
    assert q["cells"]*((r0+1/r0)/2-1) >= (r+1/r)/2-1
    assert q["cells"]*(r0-1/r0) >= r-1/r


def test_error_recurrence_refuses_overbudget_grid():
    q = choose_axis({"ellipse": 4., "hybrid_halfwidth": 1.}, 1., 1e-4)
    with pytest.raises(ValueError, match="Accumulated"):
        grid_plan([q]*6, 1e-30)


def test_single_axis_domains_and_all_time_sensitivity(proof):
    observer = CircularObserver()
    certificate = axis_neighborhood(proof, observer, 0, 2.)
    phase = phase_bound(proof, certificate, observer, [-.125, .25], [1/ARCSEC]*2)
    assert phase["all_times_in_interval_covered"]
    assert certificate["contraction_bound"] < 1
    # Independent numerical derivative witnesses at interior states/times.
    rng = np.random.default_rng(51)
    c, w = np.array(proof["parent_center"]), np.array(proof["parent_widths"])
    from research.kepler_transform.cascade_domain import solve_hybrid
    rebase = lambda q: observer.chart(0.).rebase(q, .125, observer.chart(.125))
    for fixed in [(), (3, 4), (0, 2, 3, 4, 5)]:
        hybrid = np.array(proof["child_center"])[None]+rng.uniform(-.2,.2,(4,6))*proof["child_widths"]
        for a in fixed:
            hybrid[:,a] = c[a]+rng.uniform(-.2,.2,4)*w[a]
        eps = 1e-4
        plus, minus = hybrid.copy(), hybrid.copy()
        plus[:,0] += eps*w[0]
        minus[:,0] -= eps*w[0]
        xp,_ = solve_hybrid(proof, plus, fixed, rebase, tolerance=1e-13)
        xm,_ = solve_hybrid(proof, minus, fixed, rebase, tolerance=1e-13)
        for t in [-.125, .07, .25]:
            ap = observer.chart(0.).rebase(xp,t,observer.chart(t))[:,[3,4]]
            am = observer.chart(0.).rebase(xm,t,observer.chart(t))[:,[3,4]]
            derivative = np.sum(abs(ap-am),axis=1)/(2*eps*ARCSEC)
            assert np.max(derivative) < phase["phase_derivative_bound"]


def test_anisotropic_cost_matches_materialized_cascade():
    c, w = np.zeros(6), np.ones(6)
    parent = TensorGrid(c,w,(1,2,1,1,1,1),(2,2,2,3,2,2))
    child = TensorGrid(c,w,(2,1,1,1,1,1),(3,2,2,2,2,2))
    operator = CascadeTransfer(parent,child,lambda q:q)
    cost = transfer_cost(list(parent.shape),list(child.shape),list(child.orders))
    assert cost["stored_entries"] == operator.entries
    assert cost["intermediate_nodes"] == [m.shape[0] for m in operator.stages]
    assert cost["rebase_state_evaluations_upper_bound"] >= operator.geometry_points


def test_refusal_before_allocating_full_size_hierarchy():
    observer = CircularObserver()
    *_, c, w = domain("near")
    plan = LocalHierarchyPlan(observer,c,w,[-1e-7,1e-7],[1e-5/ARCSEC]*2,
                              tolerance=1e-4,neighborhood_ellipses=(2.,8.,32.))
    assert plan.error <= 1e-4
    assert plan.stats["tree_nodes"] == 3
    with pytest.raises(MemoryError,match="plan retained"):
        plan.materialize([[[0.,0.]]]*2,max_nodes=1)
    # Small, genuine Kepler hierarchy; full source-to-query independent check.
    frequencies = [np.array([[1e-5/ARCSEC, -1e-5/ARCSEC]])]*2
    transform = plan.materialize(frequencies,max_nodes=2_000_000,max_entries=30_000_000)
    assert transform.stats["actual_transfer_entries"] == plan.stats["total_transfer_entries"]
    data = np.array([1+2j, -.3+.7j])
    values = transform.transform(data)
    rng=np.random.default_rng(3)
    queries=c+rng.uniform(-.8,.8,(5,6))*w
    actual=transform.evaluate(values,queries)
    exact=sum(np.exp(1j*observer.chart(0.).rebase(queries,t,observer.chart(t))[:,[3,4]] @ frequencies[i][0])*data[i]
              for i,t in enumerate(plan.times))
    assert max(abs(actual-exact))/sum(abs(data)) <= plan.error+2e-12
    y=rng.normal(size=len(values))+1j*rng.normal(size=len(values))
    assert abs(np.vdot(y,values)-np.vdot(transform.adjoint(y),data)) < 1e-9


def test_replay_rejects_tampered_node_count(proof):
    from research.kepler_transform.verify_local_resolution import replay_grid
    observer = CircularObserver()
    plan = local_grid(proof, observer, [.125,.125], [0.,0.],
                      neighborhood_ellipses=(1.2,))
    assert plan['nodes'] == 1
    replay_grid(plan, proof, observer)
    plan['nodes'] += 1
    with pytest.raises(ValueError,match='Grid certificate/count'):
        replay_grid(plan, proof, observer)


@pytest.mark.parametrize('kwargs', [{'multiples': (1,)*5}, {'multiples': (1,1,1,1,1,0)},
                                    {'tolerance': 0}, {'max_order': 65}])
def test_invalid_grid_budgets_fail_before_geometry(proof, kwargs):
    with pytest.raises(ValueError):
        local_grid(proof,CircularObserver(),[.125,.125],[1.,1.],**kwargs)


def test_restart_cache_reuses_only_matching_observer_and_charges_work(tmp_path):
    *_,c,w = domain('near')
    kwargs=dict(tolerance=1e-5,neighborhood_ellipses=(2.,),cache_directory=tmp_path)
    first=LocalHierarchyPlan(CircularObserver(),c,w,[-1e-5,1e-5],[0.,0.],**kwargs)
    second=LocalHierarchyPlan(CircularObserver(),c,w,[-1e-5,1e-5],[0.,0.],**kwargs)
    assert first.cached_grids == 0
    assert second.cached_grids == 3
    assert second.stats['planning_seconds_charging_cached_work'] >= second.cached_planning_seconds
    assert first.error == second.error
    changed=LocalHierarchyPlan(CircularObserver(rate=.01),c,w,[-1e-5,1e-5],[0.,0.],**kwargs)
    assert changed.cached_grids == 0
