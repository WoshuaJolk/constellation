"""Replay hashes, saved full score vectors, and independent pixel spot checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..factorization import ARCSEC
from .fields import CommonDetector
from .plan import DetectorTimePlan, domain_source_box, statistic_intervals
from .scaling import real


def verify(directory):
    report=json.loads((directory/"results.json").read_text())
    root=Path(__file__).resolve().parents[3]
    for name,digest in report["source_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Source hash mismatch: {name}")
    for name,digest in report["artifact_sha256"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    base=Path(__file__).parent/"results/dense-1024"
    for name,digest in report["input_sha256"].items():
        path=base/name if report["kind"]=="synthetic" else root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Input hash mismatch: {name}")
    with np.load(directory/"queries.npz") as f:
        q=f["queries"]
    with np.load(directory/"scores.npz") as f:
        arrays={k:f[k] for k in f.files}
    nbound,ibound=report["bounds"]
    for row in report["measurements"]:
        count=row["queries"]
        an,ai=arrays[f"fast_n_{count}"],arrays[f"fast_i_{count}"]
        dn,di=arrays[f"direct_n_{count}"],arrays[f"direct_i_{count}"]
        assert an.shape==ai.shape==dn.shape==di.shape==(count,)
        assert np.max(abs(an-dn))<=nbound+1e-8
        assert np.max(abs(ai-di))<=ibound+1e-9
        for actual,stored in zip([np.max(abs(an-dn)),np.max(abs(ai-di))],row["max_errors"]):
            assert np.isclose(actual,stored,atol=1e-14,rtol=1e-12)
        interval=statistic_intervals(an.real,ai.real,nbound,ibound)
        mask=interval["positive_information_resolved"]
        reference=dn[mask]/np.sqrt(di[mask])
        assert np.all(reference>=interval["snr_interval"][mask,0]-1e-8)
        assert np.all(reference<=interval["snr_interval"][mask,1]+1e-8)
        assert row["partial_cache"]["persistent_bytes"]<=report["cache_budget_bytes"]
    count=max(r["queries"] for r in report["measurements"])
    indices=np.unique(np.r_[np.arange(65),np.linspace(0,count-1,32,dtype=int)])
    fresh_proofs=0
    if report["kind"]=="synthetic":
        with np.load(base/"inputs.npz") as f:
            a={k:f[k] for k in f.files}
        old=json.loads((base/"results.json").read_text())
        d=CommonDetector(a["variances"],a["sigma"],a["matrices"],offsets=a["offsets"],
                         masks=a["masks"],flux_scales=a["flux_scales"],source_half_width=26.,
                         max_bytes=4_000_000_000)
        cover=domain_source_box("near")
        assert np.all(np.array(cover["relative_reference_position_bound_arcsec"])<=26.)
        for name,side in [("time_proof",old["workload"]["retained_fourier_side"]),
                          ("information_time_proof",old["workload"]["information_fourier_side"])]:
            proof=report[name]
            p=DetectorTimePlan("near",a["times"],(side,side),d.spacing,
                               tolerance=proof["requested_tolerance"])
            assert p.order==proof["order"]
            assert p.proof["exact_arithmetic_kernel_error_bound"]<=proof["exact_arithmetic_kernel_error_bound"]
            fresh_proofs+=1
        pos=(p.positions(q[indices],a["times"])-p.center_input)/ARCSEC
        dn,di=d.direct(a["images"],pos)
    else:
        ctx=real(65,root/"research/kepler_transform/chart_results/real_inputs")
        dn,di=ctx.detector.direct(ctx.images,ctx.positions(q[indices]))
    assert np.max(abs(dn-arrays[f"direct_n_{count}"][indices]))<1e-8
    assert np.max(abs(di-arrays[f"direct_i_{count}"][indices]))<1e-8
    return {"kind":report["kind"],"full_vectors_checked":2*len(report["measurements"]),
            "independent_pixel_queries_recomputed":len(indices),
            "time_proofs_replayed":fresh_proofs,"source_input_and_artifact_hashes_checked":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path)
    args=parser.parse_args()
    for directory in args.directories:
        print(json.dumps(verify(directory)),flush=True)
