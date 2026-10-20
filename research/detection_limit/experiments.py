"""Run full-objective attempts and preserve both successful and failed checks.

OPENBLAS_NUM_THREADS=1 python -m research.detection_limit.experiments NEW.json
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import i0e
from scipy.stats import norm, poisson

from .certificates import (GaussianCover,certify,certify_likelihood,geometric_detector,
                           least_favorable_weights,flux_bracket)
from .exact_controls import circle_power
from .interval_orbits import MU,Exposure,orbit_cover
from research.kepler_transform.universal import apparent as approximate_apparent


# Fixed after the pilot run; fresh streams keep validation independent of the
# pilot results used to choose this larger budget. Do not search for good seeds.
VALIDATION_SEED=92026


def exact_cases():
    reports=[]
    for alpha,signal in [(.01,3.),(1e-7,3.),(1e-7,6.)]:
        draw=lambda rng,n: signal*(rng.normal(size=n)+signal)-signal**2/2
        report=certify_likelihood(draw,alpha=alpha,samples=4194304,training_samples=1048576,seed=VALIDATION_SEED)
        report["independent_exact_power"]=float(norm.sf(norm.isf(alpha)-signal))
        report["model"]={"kind":"known Gaussian source","mean_norm":signal}
        reports.append(report)
    def circle_draw(rng,n):
        y=rng.normal(size=(n,2));y[:,0]+=3
        z=3*np.linalg.norm(y,axis=1)
        return -4.5+np.log(i0e(z))+z
    circle=certify_likelihood(circle_draw,samples=4194304,training_samples=1048576,seed=VALIDATION_SEED)
    circle["independent_exact_result"]=circle_power(3)
    circle["model"]={"kind":"continuous circle of means","radius":3,
                     "interpretation":"classical symmetry control, not an orbital source family"}
    reports.append(circle)
    for alpha in [.01,1e-7]:
        background,signal=20.,10.
        k=int(poisson.isf(alpha,background))
        boundary=(alpha-poisson.sf(k,background))/poisson.pmf(k,background)
        exact=poisson.sf(k,background+signal)+boundary*poisson.pmf(k,background+signal)
        draw=lambda rng,n:rng.poisson(background+signal,n)*np.log1p(signal/background)-signal
        report=certify_likelihood(draw,alpha=alpha,samples=4194304,training_samples=1048576,seed=VALIDATION_SEED)
        report["independent_exact_power"]=float(exact)
        report["model"]={"kind":"known Poisson source","background":background,"source_counts":signal}
        reports.append(report)
    return reports


def minimax_cases():
    means=np.array([[2.2,0.],[0.,2.9]])
    cover=GaussianCover(means,np.zeros(2),[.5,.5])
    weights,optimization=least_favorable_weights(means,samples=262144)
    uniform=certify(cover,objective="minimax",samples=1048576,training_samples=131072,seed=VALIDATION_SEED)
    optimized=certify(cover,objective="minimax",detector_weights=weights,
                     samples=2097152,training_samples=262144,seed=VALIDATION_SEED)
    # Nonzero balls cover every mean in two continuous neighborhoods, not just
    # their centers. The same weights remain only a candidate, never a proof.
    continuous=certify(GaussianCover(means,[.001,.001],[.5,.5]),objective="minimax",
                       detector_weights=weights,samples=2097152,training_samples=262144,seed=VALIDATION_SEED)
    return {"means":means.tolist(),"optimization":optimization,"weights":weights.tolist(),
            "uniform_weights":uniform,"optimized_weights":optimized,
            "continuous_mean_balls":continuous,"mean_ball_radius":.001}


def exposures_for(state, times=(-1.5,-.5,.5,1.5), shape=(7,7)):
    times=np.asarray(times,float)
    phase=2*np.pi*times/365.25
    observers=np.c_[np.cos(phase),np.sin(phase),np.zeros(len(times))]
    # Approximate geometry only CHOOSES the input WCS. The actual forward
    # model/cover is evaluated independently using interval dynamics.
    positions,_=approximate_apparent(np.asarray(state)[None,:],times,observers)
    exposures=[]
    for j,t in enumerate(times):
        u=positions[0,j,:3]-observers[j];u/=np.linalg.norm(u)
        east=np.cross([0.,0.,1.],u);east/=np.linalg.norm(east)
        north=np.cross(u,east)
        exposures.append(Exposure(float(t),tuple(observers[j]),tuple(u),tuple(east),tuple(north),shape=shape))
    return exposures


def resolved_multimodal_orbits():
    state=np.array([3.,0,0,0,np.sqrt(float(MU)/3),0])
    exposures=exposures_for(state,shape=(7,39))
    means,radii,centers=[],[],[]
    start=time.perf_counter()
    half=[2e-9,2e-9,2e-9,2e-11,2e-11,2e-11]
    for arcsec in np.linspace(-15,15,16):
        center=state.copy();center[1]=2*np.tan(arcsec*np.pi/648000)
        cover=orbit_cover(center,half,[0]*6,exposures,6.,light_time_max=.1)
        means.append(cover.means[0]);radii.append(cover.radii[0]);centers.append(center.tolist())
    cover=GaussianCover(np.array(means),np.array(radii),np.ones(16))
    construction=time.perf_counter()-start
    compressed,compression=cover.compressed(1e-8)
    # Save the executable reduced means and mapping, as well as original inputs.
    result=certify(compressed,samples=4194304,training_samples=1048576,seed=VALIDATION_SEED)
    bright=GaussianCover(compressed.means*2,compressed.radii*2,compressed.weights)
    result["bright_source_calibration_probe"]={
        method:certify(bright,samples=262144,training_samples=131072,seed=VALIDATION_SEED+2,
                       calibration=method) for method in ("alternative","null_order_statistic")}
    unit_flux=GaussianCover(compressed.means/6,compressed.radii/6,compressed.weights)
    result["brightness_limit"]=flux_bracket(unit_flux,maximum_flux=12.,samples=262144,
                                           seed=VALIDATION_SEED+1,calibration="null_order_statistic")
    result.update({"model":"uniform prior on 16 disjoint six-dimensional orbit boxes",
                   "centers":centers,"half_widths":half,"flux":6.,
                   "exposures":[asdict(e) for e in exposures],
                   "construction_seconds":construction,
                   "orbit_cell_exposure_evaluations":16*len(exposures),
                   "mean_error_radii":cover.radii.tolist(),
                   "score_means":compressed.means.tolist(),"compression":compression,
                   "between_mode_spacing_arcsec":2.,
                   "within_mode_transverse_half_width_arcsec_approx":half[1]/2*648000/np.pi,
                   "scope":"Resolved alternatives, but very narrow uncertainty within each mode; not a blind continuous search"})
    return result


def broad_orbits():
    cases=[]
    for label,radius,speed_factor in [("bound",3.,1.),("near_parabolic",3.,np.sqrt(2)),
                                       ("unbound",3.,1.8),("close_observer",1.02,1.)]:
        state=np.array([radius,0,0,0,speed_factor*np.sqrt(float(MU)/radius),0])
        exposures=exposures_for(state)
        half=[1e-3,2e-5,2e-5,1e-5,3e-6,3e-6]
        refinements=[[0]*6,[0,2,2,0,1,1],[0,3,3,0,1,1]]
        if label=="bound":
            refinements.append([1,4,4,1,1,1])
        for levels in refinements:
            start=time.perf_counter()
            result={"case":label,"center":state.tolist(),"half_widths":half,"levels":levels,
                    "flux":6.,"exposures":[asdict(e) for e in exposures]}
            try:
                cover=orbit_cover(state,half,levels,exposures,6.,light_time_max=.1)
                result.update({"cell_count":len(cover.means),"construction_seconds":time.perf_counter()-start,
                               "minimum_tv":float(np.min(cover.tv)),"maximum_tv":float(np.max(cover.tv)),
                               "prior_average_tv":float(cover.weights@cover.tv),
                               "minimum_mean_radius":float(np.min(cover.radii)),
                               "maximum_mean_radius":float(np.max(cover.radii)),
                               "orbit_cell_exposure_evaluations":len(cover.means)*len(exposures)})
                compressed,metadata=cover.compressed(1e-7)
                result["score_dimension"]=metadata["score_dimension"]
                # A small validation budget is sufficient to demonstrate that
                # these large covering errors cannot meet the requested gap.
                result["certificate"]=certify(compressed,samples=16384,training_samples=8192,seed=VALIDATION_SEED)
                result["gap_target_met"]=result["certificate"]["gap_target_met"]
            except (ValueError,ArithmeticError) as error:
                result.update({"construction_seconds":time.perf_counter()-start,
                               "failure":str(error),"gap_target_met":False})
            print(f"  broad {label}, {2**sum(levels)} cells: gap target {result['gap_target_met']}",flush=True)
            cases.append(result)
    return cases


def nuisance_cases():
    cover=GaussianCover([[3.,0.]],[0.001],[1.])
    results=[]
    for null_tv in [0.,1e-5,1e-3,.02]:
        try:
            result=certify(cover,null_tv=null_tv,alternative_tv=1e-4,samples=1048576,training_samples=131072,seed=VALIDATION_SEED)
        except ValueError as error:
            result={"null_tv_radius":null_tv,"failure":str(error),"gap_target_met":False}
        results.append(result)
    return results


def geometry_cases():
    convex=GaussianCover([[3.,-2.],[3.,0.],[3.,2.]],[0,0,0],[1,1,1])
    antipodes=GaussianCover([[3.,0.],[-3.,0.]],[0,0],[1,1])
    return {"convex_affine_control":geometric_detector(convex),
            "nonconvex_antipodes":geometric_detector(antipodes),
            "antipodes_exact_minimax_power":float(norm.sf(norm.isf(.005)-3)+norm.cdf(-norm.isf(.005)-3)),
            "interpretation":"Convex separation can be exact; convexifying physical alternatives can destroy useful power"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    result={"seed":VALIDATION_SEED,"requested_gap":.01,
            "confidence_scope":"Each stochastic certificate separately has 99% confidence; no experiment-wide joint confidence claimed",
            "software":{"python":platform.python_version(),**{p:importlib.metadata.version(p) for p in ["numpy","scipy","python-flint"]}},
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}}
    for name,function in [("exact_controls",exact_cases),("minimax",minimax_cases),
                          ("resolved_multimodal_orbits",resolved_multimodal_orbits),
                          ("broad_orbit_families",broad_orbits),("nuisance_uncertainty",nuisance_cases),
                          ("geometry",geometry_cases)]:
        print(name,flush=True);start=time.perf_counter()
        result[name]=function()
        result.setdefault("timings_seconds",{})[name]=time.perf_counter()-start
    requirements={
        "exact_control_gap_targets_met":all(r["gap_target_met"] for r in result["exact_controls"]),
        "multimodal_continuous_orbit_gap_target_met":result["resolved_multimodal_orbits"]["gap_target_met"],
        "multimodal_brightness_limit_bracket_target_met":result["resolved_multimodal_orbits"]["brightness_limit"]["gap_target_met"],
        "continuous_minimax_control_gap_target_met":result["minimax"]["continuous_mean_balls"]["gap_target_met"],
        "broad_resolved_orbit_gap_targets_met":all(r["gap_target_met"] for r in result["broad_orbit_families"]),
        "general_instrument_noise_and_nuisance_models":False,
        "complete_statistical_roundoff_certification":False,
        "independent_novelty_and_specialist_review":False,
    }
    result["assessment"]={"requirements":requirements,"full_objective_met":all(requirements.values()),
                          "breakthrough_claim":False}
    with args.output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result["assessment"],indent=2),flush=True)


if __name__=="__main__":main()
