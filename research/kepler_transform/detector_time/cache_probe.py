"""Measure the larger-memory direct factor cache on an existing query set."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..factorization import ARCSEC
from ..temporal.transform import TimePlan
from .benchmark import save
from .fields import CommonDetector
from .pixel_cache import PixelCache
from .plan import DetectorTimePlan
from .scaling import timed_values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--queries",type=int,default=8192)
    parser.add_argument("--cache-bytes",type=int,default=6_000_000_000)
    args=parser.parse_args()
    source=Path(__file__).parent/"results/dense-1024/inputs.npz"
    with np.load(source) as f:
        a={k:f[k] for k in f.files}
    with np.load(args.campaign/"queries.npz") as f:
        q=f["queries"][:args.queries]
    if len(q)!=args.queries:
        raise ValueError("Insufficient saved queries")
    warm=TimePlan("near",[-1e-7,1e-7])
    warm.positions(q[:1],[-4.,4.])
    # This probe uses only exact direct positions, never the time interpolation.
    p=DetectorTimePlan("near",a["times"],(3,3),.01)
    d=CommonDetector(a["variances"],a["sigma"],a["matrices"],offsets=a["offsets"],
                     masks=a["masks"],flux_scales=a["flux_scales"],source_half_width=26.,
                     max_bytes=4_000_000_000)
    start=perf_counter()
    pos=(p.positions(q,a["times"])-p.center_input)/ARCSEC
    geometry=perf_counter()-start
    start=perf_counter()
    cache=PixelCache(d,pos,backend="factor",max_bytes=args.cache_bytes)
    setup=perf_counter()-start
    (n,i), timing=timed_values(lambda:np.array(cache.apply(a["images"])),3)
    with np.load(args.campaign/"scores.npz") as f:
        dn,di=f[f"direct_n_{len(q)}"],f[f"direct_i_{len(q)}"]
    error=float(max(np.max(abs(n-dn)),np.max(abs(i-di))))
    if error>1e-8:
        raise ValueError("Large cache does not agree with stored direct scores")
    changed=.7*a["images"]+np.random.default_rng(118).normal(size=a["images"].shape)*.01
    cn,ci=cache.apply(changed)
    indices=np.unique(np.r_[np.arange(65),np.linspace(0,len(q)-1,32,dtype=int)])
    check=d.direct(changed,pos[indices])
    changed_error=float(max(np.max(abs(cn[indices]-check[0])),np.max(abs(ci[indices]-check[1]))))
    if changed_error>1e-8:
        raise ValueError("Changed-image direct cache check failed")
    record={"queries":len(q),"cache_budget_bytes":args.cache_bytes,
            "persistent_cache_bytes":cache.persistent_bytes,"geometry_seconds":geometry,
            "cache_setup_seconds":setup,"new_image_application":timing,
            "max_score_error":error,"changed_image_max_score_error":changed_error,
            "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [source,args.campaign/"queries.npz",args.campaign/"scores.npz"]},
            "source_sha256":{str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "scope":"Memory sensitivity control, not a new compressed-pipeline timing.",
            "cache_budget_excludes_shared_inputs_and_transient_arrays":True}
    save(args.output,record)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
