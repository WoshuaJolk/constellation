"""Replay certificates and independently check exact subdivision coverage."""
import argparse
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations
import json
from math import prod
from pathlib import Path

import numpy as np
from flint import arb, ctx

from magictools.ejector_bounds import bound_family, refine_group, certify_witness, outward_float_bounds


def read(path):
    if path.suffix == '.gz':
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def key(ids): return tuple(sorted(ids))


def check_cover(refinement, cost):
    """Independent exact rational volume + disjoint-interior covering check."""
    leaves = refinement['terminal_leaves']+refinement['pending_boxes']
    boxes = [np.asarray(r['bounds'], float) for r in leaves]
    if not boxes: raise AssertionError('A refinement must retain a covering partition')
    old = ctx.prec; ctx.prec = 128
    try: _, radius = outward_float_bounds(arb(cost).sqrt())
    finally: ctx.prec = old
    dimension = 4*len(refinement['star_ids'])
    for b in boxes:
        assert b.shape == (dimension, 2)
        assert np.all(b[:, 0] >= -radius) and np.all(b[:, 1] <= radius)
        assert np.all(b[:, 0] < b[:, 1])
    for a, b in combinations(boxes, 2):
        assert not np.all(np.maximum(a[:, 0], b[:, 0]) < np.minimum(a[:, 1], b[:, 1])), 'Overlapping leaf interiors'
    volumes = [prod(Fraction.from_float(float(hi))-Fraction.from_float(float(lo)) for lo, hi in b) for b in boxes]
    assert sum(volumes) == (2*Fraction.from_float(float(radius)))**dimension, 'Leaf volumes do not cover root'


def verify(root, case):
    spec = read(root/(case+'.json'))
    certificate = read(root/(case+'-bounds')/'complete.json')
    search_result = read(root/case/'complete.json')
    fit_path = root/case/'observed_fits.json.gz'
    if not fit_path.exists(): fit_path = root/case/'observed_fits.json'
    observed_fits = read(fit_path)
    control_bundle = root/case/'control_fits.json.gz'
    controls = read(control_bundle) if control_bundle.exists() else [read(p) for p in sorted((root/case).glob('control_*.json'))]
    coarse, enclosures = bound_family(spec)
    with np.load(root/(case+'-bounds')/'enclosures.npz') as saved:
        assert np.array_equal(enclosures, saved['expanded_position_bounds_pc'])
    family = {key(r['id'] for r in group) for group in combinations(spec['objects'], spec['search']['group_size'])}
    assert len(controls) == spec['controls']['n_controls']
    for ledger, expected in zip([observed_fits]+controls, [search_result['minimum_chi2']]+search_result['controls']['minimum_chi2']):
        assert len(ledger) == len(family)
        actual = {key(spec['objects'][i]['id'] for i in row['indices']) for row in ledger}
        assert actual == family
        values = [r['chi2'] for r in ledger if r['chi2'] is not None]
        assert (min(values) if values else None) == expected
    negative = {key(g) for g in certificate['excluded_groups']}
    positive = {key(w['star_ids']) for w in certificate['certified_witnesses']}
    unresolved = {key(g['star_ids']) for g in certificate['unresolved_groups']}
    assert not negative & positive and not negative & unresolved and not positive & unresolved
    assert negative | positive | unresolved == family
    initially_negative = {key(g) for g in coarse['excluded_groups']}
    assert initially_negative <= negative
    refined_negative = set()
    for refinement in certificate['refinements']:
        check_cover(refinement, spec['search']['report_chi2_max'])
        replay = refine_group(spec, refinement['star_ids'], refinement['max_nodes'])
        assert replay == refinement
        if refinement['excluded']: refined_negative.add(key(refinement['star_ids']))
    assert negative == initially_negative | refined_negative
    for witness in certificate['certified_witnesses']:
        fitted = dict(events=[dict(lookback_myr=witness['lookback_myr'])], fitted_objects=witness['fitted_objects'])
        checked = certify_witness(spec, witness['star_ids'], fitted)
        assert checked['certified']
        assert checked['total_mahalanobis_cost_upper'] == witness['total_mahalanobis_cost_upper']
        assert checked['centroid_radius_upper_pc'] == witness['centroid_radius_upper_pc']
    for row in certificate['provenance']['inputs']:
        path = Path(row['path'])
        if not path.is_absolute(): path = root/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
    result = dict(case=case, hypotheses=len(family), excluded=len(negative), certified_witnesses=len(positive),
                  unresolved=len(unresolved), all_claims_replayed=True, subdivision_coverage_checked_with_exact_rationals=True)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--case', choices=['observed', 'injection', 'historical'], action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = [verify(args.directory, case) for case in (args.case or ['observed', 'injection', 'historical'])]
    if args.output: args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
