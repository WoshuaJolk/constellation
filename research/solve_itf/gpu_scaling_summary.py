"""Read-only comparisons of matched scaling runs; no new scientific filtering."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np


def rows(path):
    with gzip.open(path,'rt') as f:r=json.load(f)
    return {tuple(x['tracklet_indices']):x for x in r['candidates']}


def compare(a,b):
    shared=a.keys()&b.keys();only_a=sorted(a.keys()-b.keys());only_b=sorted(b.keys()-a.keys())
    exact=sum(a[k]==b[k] for k in shared)
    rms=max((abs(a[k]['modes'][0]['rms_arcsec']-b[k]['modes'][0]['rms_arcsec']) for k in shared),default=0.)
    return dict(left_count=len(a),right_count=len(b),shared=len(shared),identical_complete_rows=exact,
        left_only=only_a,right_only=only_b,max_best_rms_difference_arcsec=rms,
        mode_count_differences=sum(len(a[k]['modes'])!=len(b[k]['modes']) for k in shared))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--followup',type=Path);args=p.parse_args();root=args.root
    report=json.loads((root/'report.json').read_text());inv=json.loads((root/'inventory.json').read_text())
    phases=report['phases'];summary=dict(status=report['status'],available_distinct_seeds=inv['available_seeds'],
        requested_sizes=[100000,300000,1000000],actual_sizes=report['sizes'],shortfall=report['shortfall'],
        shared_graph_seconds=inv['total_seconds'],phases=[],comparisons=[])
    for phase in phases:
        s=json.loads((root/phase['name']/'summary.json').read_text());receipts=s['batch_receipts']
        summary['phases'].append({k:v for k,v in s.items() if k!='batch_receipts'}|dict(
            aggregate=dict(chains=sum(r['chains'] for r in receipts),starts=sum(r['starts'] for r in receipts),
                planning_cpu_seconds=sum(r['planning_seconds'] for r in receipts),
                gpu_solve_wall_seconds=sum(r['gpu_solve_profile']['wall_seconds'] for r in receipts),
                gpu_setup_seconds=sum(r['gpu_setup_seconds'] for r in receipts),
                selection_cpu_seconds=sum(r['selection_seconds'] for r in receipts),
                verify_cpu_seconds=sum(r['cpu_verify_fallback_seconds'] for r in receipts),
                fallback_starts=sum(r['cpu_fallback_starts'] for r in receipts))))
    for size in report['sizes']:
        available={p['backend']:p for p in summary['phases'] if p['actual_seeds']==size and p['status']=='bounded_search_complete'}
        if 'stream' not in available:continue
        new=available['stream'];new_rows=rows(root/new['name']/'report.json.gz')
        for backend in ['cpu','prefetch']:
            if backend not in available:continue
            old=available[backend]
            if old['seed_sha256']!=new['seed_sha256']:raise ValueError('Mismatched seed cohort')
            compared=compare(rows(root/old['name']/'report.json.gz'),new_rows)
            summary['comparisons'].append(dict(seeds=size,left=backend,right='stream',
                cached_graph_speedup=old['application_seconds']/new['application_seconds'],
                including_graph_build_speedup=old['with_shared_graph_build_seconds']/new['with_shared_graph_build_seconds'],**compared))
    if args.followup:
        follow=json.loads((args.followup/'report.json').read_text())
        summary['followup_status']=follow['status']
        for phase in follow.get('phases',[]):
            name=phase['name'];s=json.loads((args.followup/name/'summary.json').read_text());a=s['batch_receipts']
            summary['phases'].append({k:v for k,v in s.items() if k!='batch_receipts'}|dict(backend='streamv2',aggregate=dict(
                chains=sum(v['chains'] for v in a),starts=sum(v['starts'] for v in a),
                planning_cpu_seconds=sum(v['planning_seconds'] for v in a),
                gpu_solve_wall_seconds=sum(v['gpu_solve_profile']['wall_seconds'] for v in a),
                gpu_setup_seconds=sum(v['gpu_setup_seconds'] for v in a),
                selection_cpu_seconds=sum(v['selection_seconds'] for v in a),
                verify_cpu_seconds=sum(v['cpu_verify_fallback_seconds'] for v in a),
                fallback_starts=sum(v['cpu_fallback_starts'] for v in a))))
            if s['status']!='bounded_search_complete':continue
            current=rows(args.followup/name/'report.json.gz')
            matched=False
            for prior in summary['phases']:
                if prior['backend']=='streamv2' or prior['actual_seeds']!=s['actual_seeds'] or prior['status']!='bounded_search_complete':continue
                assert prior['seed_sha256']==s['seed_sha256']
                compared=compare(rows(root/prior['name']/'report.json.gz'),current);matched=True
                summary['comparisons'].append(dict(seeds=s['actual_seeds'],left=prior['backend'],right='streamv2',
                    cached_graph_speedup=prior['application_seconds']/s['application_seconds'],
                    including_graph_build_speedup=prior['with_shared_graph_build_seconds']/s['with_shared_graph_build_seconds'],**compared))
            if not matched:summary.setdefault('no_complete_reference_comparison',[]).append(name)
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['phases','comparisons']}))
    for p in summary['phases']:print(p['name'],p['application_seconds'],p['candidates'])
    for c in summary['comparisons']:print({k:v for k,v in c.items() if k not in ['left_only','right_only']})


if __name__=='__main__':main()
