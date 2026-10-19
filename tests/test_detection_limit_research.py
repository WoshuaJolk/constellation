"""Independent checks of the research certificates and interval forward model."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ndtr, i0e
from scipy.stats import ncx2, norm, poisson

from research.detection_limit.certificates import (
    GaussianCover, certify, certify_likelihood, gaussian_tv, geometric_detector,
    least_favorable_weights, log_lr, rejection_probability,
)

flint=pytest.importorskip("flint")
from flint import arb, ctx
from research.detection_limit.interval_orbits import (
    MU,C_AU_DAY,Exposure,apparent,propagate,step,mean_ball,uniform_boxes,
    pixel_means,gaussian_pixel,orbit_cover,
)
from research.detection_limit.exact_controls import circle_power


def test_tv_and_cover_validation():
    assert gaussian_tv(0)==0
    assert gaussian_tv(2)==pytest.approx(2*norm.cdf(1)-1)
    with pytest.raises(ValueError):
        GaussianCover([[1,0]],[-1],[1])
    with pytest.raises(ValueError):
        certify(GaussianCover([[1,0]],[0],[1]),null_tv=.02,alpha=.01)


def test_mixture_likelihood_against_direct_normal_densities():
    means=np.array([[1.,2.],[-.5,3.]])
    y=np.array([[0.,1.],[2.,-3.]])
    weights=np.array([.3,.7])
    ratio=(np.exp(-np.sum((y[:,None,:]-means)**2,axis=2)/2)@weights)/np.exp(-np.sum(y*y,axis=1)/2)
    np.testing.assert_allclose(np.exp(log_lr(y,means,weights)),ratio,rtol=3e-15)


@pytest.mark.parametrize("alpha",[.01,1e-7])
def test_importance_certificate_contains_exact_known_source_power(alpha):
    cover=GaussianCover([[3.,0.]],[0],[1])
    report=certify(cover,alpha=alpha,samples=131072,training_samples=32768)
    exact=norm.sf(norm.isf(alpha)-3)
    lo,hi=report["optimal_power_interval"]
    assert lo<=exact<=hi
    assert report["null_probability_upper"]<=alpha+1e-15
    assert 0<=rejection_probability([5,0],cover.means,report)<=1


def test_continuous_circle_exact_and_simulated_certificate():
    exact=circle_power(3)
    reference=ncx2.sf(-2*np.log(.01),2,9)
    lo,hi=exact["power_interval"]
    assert lo-2e-15<=reference<=hi+2e-15
    assert hi-lo<1e-14
    def draw(rng,n):
        y=rng.normal(size=(n,2));y[:,0]+=3
        z=3*np.linalg.norm(y,axis=1)
        return -4.5+np.log(i0e(z))+z
    report=certify_likelihood(draw,samples=131072)
    assert report["optimal_power_interval"][0]<=reference<=report["optimal_power_interval"][1]


def test_poisson_boundary_randomization_closes_discrete_gap():
    background,signal=20.,10.
    alpha=.01
    threshold=int(poisson.isf(alpha,background))
    boundary=(alpha-poisson.sf(threshold,background))/poisson.pmf(threshold,background)
    exact=poisson.sf(threshold,background+signal)+boundary*poisson.pmf(threshold,background+signal)
    draw=lambda rng,n:rng.poisson(background+signal,n)*np.log1p(signal/background)-signal
    report=certify_likelihood(draw,samples=262144,training_samples=65536)
    assert report["optimal_power_interval"][0]<=exact<=report["optimal_power_interval"][1]
    assert 0<report["detector"]["boundary_acceptance_probability"]<1
    assert report["optimality_gap_upper"]<.025


def test_no_information_and_geometry_counterexample():
    cover=GaussianCover([[0.,0.]],[0],[1])
    report=certify(cover,samples=10000)
    assert report["optimal_power_interval"][0]==.01
    antipodes=GaussianCover([[3.,0.],[-3.,0.]],[0,0],[.5,.5])
    geometry=geometric_detector(antipodes)
    exact=norm.sf(norm.isf(.005)-3)+norm.cdf(-norm.isf(.005)-3)
    assert geometry["power_interval"][0]==.01
    assert exact>.6
    assert geometry["power_interval"][1]>=exact


def test_convex_mean_family_has_matching_geometry_bounds():
    # Every (3,t), -2<=t<=2: center is physical and hardest.
    cover=GaussianCover([[3,-2],[3,0],[3,2]],[0,0,0],[1,1,1])
    report=geometric_detector(cover)
    assert report["gap"]<1e-10
    assert report["power_interval"][0]==pytest.approx(norm.sf(norm.isf(.01)-3))


def test_minimax_candidate_requires_independent_certificate():
    means=np.array([[2.2,0],[0,2.9]])
    weights,optimization=least_favorable_weights(means,samples=8192)
    assert weights[0]>.6
    assert optimization["certificate"] is False
    report=certify(GaussianCover(means,[.01,.02],[1,1]),objective="minimax",
                   detector_weights=weights,samples=32768,component_samples=32768)
    assert report["optimal_power_interval"][0]<=min(norm.sf(norm.isf(.01)-np.linalg.norm(means,axis=1)))
    assert len(report["component_power_lower_bounds"])==2


def test_compression_charges_the_discarded_direction():
    cover=GaussianCover([[2.,0.,1e-5],[3.,1e-6,0.]],[.001,.002],[1,1])
    compressed,report=cover.compressed(1e-4)
    basis=np.array(report["pixel_to_score_basis"])
    residual=np.linalg.norm(cover.means-compressed.means@basis,axis=1)
    np.testing.assert_allclose(compressed.radii,cover.radii+residual,atol=1e-15)


def test_validated_flow_encloses_analytic_circle_and_parabola():
    with ctx.workprec(96):
        mu=arb(MU);radius=arb(3);omega=(mu/(radius*radius*radius)).sqrt()
        state=[radius,arb(0),arb(0),arb(0),radius*omega,arb(0)]
        for t in [-4,4]:
            result=propagate(state,t)
            angle=omega*t
            expected=[radius*angle.cos(),radius*angle.sin(),arb(0),
                      -radius*omega*angle.sin(),radius*omega*angle.cos(),arb(0)]
            assert all((a-b).contains(0) for a,b in zip(result,expected))
        q=arb(1);v=(2*mu/q).sqrt();t=arb(2)
        result=propagate([q,0,0,0,v,0],t)
        d=2*((arb(3)/2*t/(2*q*q*q/mu).sqrt()).asinh()/3).sinh()
        assert (result[0]-q*(1-d*d)).contains(0)
        assert (result[1]-2*q*d).contains(0)


@pytest.mark.parametrize("energy_factor",[.7,1.,1.4])
def test_interval_flow_matches_independent_cartesian_integration(energy_factor):
    mu=float(MU);initial=np.array([1.2,.2,.05,-.001,energy_factor*np.sqrt(2*mu/1.2),.0003])
    def rhs(t,y):return np.r_[y[3:],-mu*y[:3]/np.linalg.norm(y[:3])**3]
    for t in [-8.,8.]:
        reference=solve_ivp(rhs,[0,t],initial,method="DOP853",rtol=2e-13,atol=2e-14).y[:,-1]
        with ctx.workprec(96): result=propagate(initial,t)
        np.testing.assert_allclose([float(x.mid()) for x in result],reference,atol=3e-12,rtol=0)


def test_gaussian_pixel_range_contains_interior_and_mode():
    with ctx.workprec(96):
        coordinate=arb(0,2)
        result=gaussian_pixel(coordinate,arb(-.5),arb(.5),1)
        for x in np.linspace(-2,2,31):
            value=ndtr(.5-x)-ndtr(-.5-x)
            assert float(result.lower())-1e-15<=value<=float(result.upper())+1e-15


def test_continuous_cell_encloses_independent_retarded_pixels():
    mu=float(MU);center=np.array([3.,0,0,0,np.sqrt(mu/3),0])
    half=np.array([1e-6]*3+[1e-8]*3)
    exposure=Exposure(1.,(1,0,0),(1,0,0),(0,1,0),(0,0,1),shape=(5,5))
    # The PSF is far outside this stamp; use a WCS pointed at the moving source.
    angle=(center[4]/2)*1
    exposure.center=(float(np.cos(angle)),float(np.sin(angle)),0.)
    exposure.east=(-float(np.sin(angle)),float(np.cos(angle)),0.)
    with ctx.workprec(96):
        cell=uniform_boxes(center,half,[0]*6)[0]
        enclosure=pixel_means(cell,[exposure],10,light_time_max=.1)
        representative,radius=mean_ball(enclosure)
    rng=np.random.default_rng(123)
    for _ in range(6):
        initial=center+rng.uniform(-1,1,6)*half
        def rhs(t,y):return np.r_[y[3:],-mu*y[:3]/np.linalg.norm(y[:3])**3]
        solution=solve_ivp(rhs,[0,1.],initial,method="DOP853",rtol=2e-13,atol=2e-14,dense_output=True)
        emission=1.
        for _ in range(10):emission=1.-np.linalg.norm(solution.sol(emission)[:3]-exposure.observer)/float(C_AU_DAY)
        direction=solution.sol(emission)[:3]-exposure.observer
        scale=np.pi/648000
        x=np.dot(direction,exposure.east)/np.dot(direction,exposure.center)/scale
        y=np.dot(direction,exposure.north)/np.dot(direction,exposure.center)/scale
        edges=np.arange(6)-2.5
        gx=np.diff(ndtr(edges-x));gy=np.diff(ndtr(edges-y))
        pixels=(10*gy[:,None]*gx).ravel()
        assert np.linalg.norm(pixels-representative)<=radius+1e-8
        for value,ball in zip(pixels,enclosure):
            assert float(ball.lower())-1e-8<=value<=float(ball.upper())+1e-8


def test_interval_rejects_singular_or_uncovered_light_time():
    with pytest.raises(ValueError):step([0]*6,1)
    with pytest.raises(ValueError):apparent([40,0,0,0,.002,0],1,(1,0,0),light_time_max=.001)
    with pytest.raises(ValueError):uniform_boxes([1]*6,[1]*6,[3]*6)
