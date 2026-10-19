#!/usr/bin/env python3
"""Refine the two spectroscopic-quality survivors of the blind measured-RV run.

This is targeted posterior refinement, not a new catalogue-wide detection test.
Candidate selection and all quality exclusions are retained in vetting.json.
"""
import json
from pathlib import Path
import numpy as np
from magictools.ejector_gaia import read_gaia
from research.ejector_catalogue.run import run
from argparse import Namespace


def refine(observed, output):
    here=Path(__file__).resolve().parent
    raw=read_gaia(here/'data/gaia_fast.fits.gz')
    quality=read_gaia(here/'data/candidate_rv_quality.vot')
    snr={str(r['source_id']):float(r['rv_expected_sig_to_noise']) for r in quality}
    rows=json.loads(Path(observed).read_text())['candidates']
    kept=[]; vetting=[]
    for row in rows:
        values=[snr[x] for x in row['star_ids']]
        accept=all(x>=5 for x in values)
        vetting.append(dict(star_ids=row['star_ids'],rv_expected_snr=values,all_rv_snr_at_least_5=accept))
        if accept:kept.append(row)
    ids=sorted(set(x for r in kept for x in r['star_ids']))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'vetting.json').write_text(json.dumps(dict(groups=vetting,input_groups=len(rows),survivors=len(kept),
        caution='SNR cut is necessary quality triage, not independent velocity confirmation or calibrated post-selection significance'),indent=2)+'\n')
    raw=raw[np.isin(np.asarray(raw['source_id']).astype(str),ids)]
    source=output/'targeted_gaia.fits'
    if not source.exists():raw.write(source)
    args=Namespace(output=output,data=source,scenario='baseline',samples=512,controls=31,control_method='tangent_rotation',
                   radius=10.,lookback=3.,step=.05,integration_step=.01,prepare_only=False)
    run(args)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('observed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();refine(a.observed,a.output)
