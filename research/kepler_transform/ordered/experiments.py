"""Reproduce the stage-aware variant without changing baseline certificates."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from flint import ctx

from ..cascade_domain import certify_domain
from ..constructive_experiments import domain
from ..local_resolution_experiments import hashes as baseline_hashes
from ..reception_flow import CircularObserver
from ..factorization import ARCSEC
from .resolution import local_grid
from .hierarchy import OrderedHierarchyPlan


def hashes():
    source=baseline_hashes();root=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))+[root/'tests/test_kepler_ordered_resolution.py']
    source.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return source


def local(case):
    *_,c,w=domain(case);observer=CircularObserver()
    proof=certify_domain(observer,c,w,0.,.125)
    grid=local_grid(proof,observer,[.125,.125],[1/ARCSEC]*2,
                    neighborhood_ellipses=(1.2,2.,4.,8.,16.,32.))
    return {'case':case,'certificate':proof,'grid':grid}


def hierarchy(case):
    *_,geom,c,w=domain(case)
    plan=OrderedHierarchyPlan(CircularObserver(),c,w,geom.times,[1/ARCSEC]*2,
                              neighborhood_ellipses=(1.05,2.,4.,8.,32.),
                              progress=lambda r:print(json.dumps({'case':case,**r}),flush=True))
    return {'case':case,'plan':plan.record()}


def numerical():
    *_,c,w=domain('near');observer=CircularObserver()
    plan=OrderedHierarchyPlan(observer,c,w,[-1e-7,1e-7],[1/ARCSEC]*2,
                              neighborhood_ellipses=(2.,8.,32.))
    frequencies=[np.array([[1/ARCSEC,-1/ARCSEC]])]*2
    graph=plan.materialize(frequencies,max_nodes=2_000_000,max_entries=30_000_000)
    data=np.array([1+2j,-.3+.7j]);rng=np.random.default_rng(134)
    queries=c+rng.uniform(-.999,.999,(64,6))*w
    begin=perf_counter();values=graph.transform(data);seconds=perf_counter()-begin
    actual=graph.evaluate(values,queries)
    from ..compiled_reference import projected
    from ..segment_experiments import geometry
    positions=projected(observer.chart(0.).states(queries),geometry(plan.times,observer.axes))
    exact=sum(np.exp(1j*positions[:,i] @ frequencies[i][0])*data[i] for i in range(2))
    y=rng.normal(size=len(values))+1j*rng.normal(size=len(values))
    return {'plan':plan.record(),'materialized':graph.stats,
            'sampled_error_per_source_l1':float(max(abs(actual-exact))/sum(abs(data))),
            'query_count':len(queries),'adjoint_absolute_discrepancy':float(abs(np.vdot(y,values)-np.vdot(graph.adjoint(y),data))),
            'application_seconds':seconds,'time_span_days':2e-7,'numerical_regression_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--mode',choices=['local','hierarchy','numerical'],required=True)
    parser.add_argument('--case',choices=['near','distant','stress'],default='near')
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=False)
    ctx.prec=100;source=hashes();begin=perf_counter()
    artifact={'source_sha256':source,'arb_precision_bits':100,'full_objective_complete':False}
    artifact[args.mode]=(local(args.case) if args.mode=='local' else hierarchy(args.case) if args.mode=='hierarchy' else numerical())
    artifact['total_seconds']=perf_counter()-begin
    if hashes()!=source:raise RuntimeError('Source changed during ordered campaign')
    (args.directory/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'saved':str(args.directory),'seconds':artifact['total_seconds']}),flush=True)

if __name__=='__main__':main()
