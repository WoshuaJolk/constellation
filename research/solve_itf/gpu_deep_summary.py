"""Summarize recorded deep-benchmark receipts without running any numerical work."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def compare_records(left, right):
    """Compare frozen outputs only; do not refit or pick a favorable subset."""
    def load(path):
        with gzip.open(path, 'rt') as stream:
            return {r['source_position']: r for r in map(json.loads, stream)
                    if r['within_budget']}
    a, b = load(left), load(right)
    common = sorted(a.keys() & b.keys())
    status_changes = []
    rms_changes = []
    mode_changes = []
    max_rms_difference = 0.
    for key in common:
        x, y = a[key], b[key]
        assert x['membership'] == y['membership']
        def status(record):
            return record.get('status') or ('unavailable' if record['error'] == 'Observer geometry unavailable'
                    else 'numerical_error' if record['error']
                    else 'retained' if record['row'] is not None else 'not_retained')
        if status(x) != status(y):
            status_changes.append({'source_position': key, 'left': status(x), 'right': status(y)})
        if x['row'] is not None and y['row'] is not None:
            delta = abs(x['row']['rms_arcsec'] - y['row']['rms_arcsec'])
            max_rms_difference = max(max_rms_difference, delta)
            if delta > .001:
                rms_changes.append({'source_position': key, 'difference_arcsec': delta})
            if len(x['row']['modes']) != len(y['row']['modes']):
                mode_changes.append(key)
    return dict(common_within_budget=len(common), status_changes=status_changes,
                max_retained_rms_difference_arcsec=max_rms_difference,
                rms_changes_over_001_arcsec=rms_changes, mode_count_changes=mode_changes)


def summarize(directory):
    p=Path(directory);report=json.loads((p/'report.json').read_text());result={'status':report['status'],'setup':report.get('setup',{}),'backends':{},'comparisons':report.get('comparisons',[]),'planetary':report.get('longgap',{}),'limitations':['Frozen checkpoint is reused: no measurement of fresh 7M graph construction.','Full catalogue is available, but timed passes process only their measured subset of distinct original seed triples.','No independent truth labels for the real seed or historical planetary-mode workloads. Retained does not mean confirmed.','Both phases ran on the H100 host at its full hourly rate; this is not a standalone CPU-host cost comparison.','Planetary propagation, uncertainty retrieval and final joint fitting remain CPU-only. No GPU speedup is assigned to that stage.','Remaining-seed projections cover this fitting policy only, not a whole-campaign completion estimate.']}
    samples=defaultdict(list)
    if (p.parent/'deleted.json').exists() and (p.parent/'launch.json').exists():
        deletion = json.loads((p.parent/'deleted.json').read_text())
        launch = json.loads((p.parent/'launch.json').read_text())
        result['accounting'] = {
            'estimated_compute_usd': deletion['estimated_compute_usd'],
            'worker_hourly_usd': launch['price_hourly'],
            'worker_lifetime_seconds': deletion['deleted_utc']-launch['created_local'],
            'worker_deleted_unix': deletion['deleted_utc'],
            'remaining_droplet_ids_at_deletion': deletion['remaining_ids'],
            'scope': 'Compute estimate includes setup, transfer, tests, idle intervals and cleanup; not a provider invoice or historical storage total.'}
    with (p/'utilization.jsonl').open() as stream:
        for row in map(json.loads,stream):samples[row['phase']].append(row)
    for backend in ['cpu','gpu']:
        phases=[(name,row) for name,row in report.get('fits',{}).items() if name.startswith(backend+'-')]
        total_seconds=sum(min(row['budget_seconds'],row['elapsed_seconds']) for _,row in phases)
        counts={key:sum(row['within_budget_counts'].get(key,0) for _,row in phases) for key in ['attempted','starts','retained','not_retained','numerical_errors','unavailable','modes']}
        util=[r for name,_ in phases for r in samples[name]]
        value={'charged_comparison_seconds':total_seconds,'within_budget_counts':counts,'all_elapsed_seconds':sum(row['elapsed_seconds'] for _,row in phases),'phases':[]}
        if total_seconds:
            value['chains_per_second']=counts['attempted']/total_seconds;value['starts_per_second']=counts['starts']/total_seconds
            value['projected_7m_fitting_hours_only']=7000000/value['chains_per_second']/3600 if counts['attempted'] else None
        for name,row in phases:
            seconds=min(row['budget_seconds'],row['elapsed_seconds'])
            value['phases'].append(dict(name=name,seconds=seconds,elapsed_seconds=row['elapsed_seconds'],within_budget_counts=row['within_budget_counts'],all_counts=row['counts'],chains_per_second=row['within_budget_counts'].get('attempted',0)/seconds))
        if util:
            value['utilization']={'samples':len(util),'cpu_mean_busy_percent':float(np.average([r['cpu_busy_percent'] for r in util],weights=[r['interval_seconds'] for r in util])),'gpu_mean_util_percent':float(np.mean([r['gpu_util_percent'] for r in util if 'gpu_util_percent' in r])),'gpu_peak_memory_mib':max(r.get('gpu_memory_used_mib',0) for r in util),'sampling_note':'5-second samples; CPU interval may straddle a stage transition.'}
        if backend=='gpu':
            batches=[]
            for name,_ in phases:
                path=p/(name+'-batches.json')
                if path.exists():batches.extend(b for b in json.loads(path.read_text()) if b['within_budget'])
            value['gpu_batch_components_seconds']={k:sum(b['gpu'][k] for b in batches) for k in ['planning_seconds','gpu_setup_seconds','cpu_verify_fallback_seconds','total_seconds']}
            value['gpu_batch_components_seconds']['gpu_solve_seconds']=sum(b['gpu']['gpu_solve_profile']['wall_seconds'] for b in batches)
            components = value['gpu_batch_components_seconds']
            components['selection_and_other_assembly_seconds'] = components['total_seconds'] - sum(
                components[k] for k in ['planning_seconds', 'gpu_setup_seconds',
                                       'cpu_verify_fallback_seconds', 'gpu_solve_seconds'])
            value['gpu_batch_component_fractions'] = {k: v/components['total_seconds']
                for k, v in components.items() if k != 'total_seconds'} if components['total_seconds'] else {}
            value['cpu_fallback_starts']=sum(b['gpu']['cpu_fallback_starts'] for b in batches)
        result['backends'][backend]=value
    cpu=result['backends']['cpu'].get('chains_per_second',0);gpu=result['backends']['gpu'].get('chains_per_second',0)
    result['measured_fitting_throughput_ratio_gpu_over_cpu']=gpu/cpu if cpu else None
    distinct = set()
    for name in report.get('fits', {}):
        with gzip.open(p/(name+'.jsonl.gz'), 'rt') as stream:
            distinct.update(r['source_position'] for r in map(json.loads, stream)
                            if r['within_budget'])
    result['distinct_seed_positions_in_main_timed_passes'] = len(distinct)
    result['fraction_of_original_7m_queue_in_main_timed_passes'] = len(distinct)/7000000
    process_dir = p.parent/'process-results'
    if (process_dir/'report.json').exists():
        process = json.loads((process_dir/'report.json').read_text())
        value = {k: process[k] for k in ['status', 'setup_seconds', 'budget_seconds',
                 'elapsed_seconds', 'counts', 'within_budget_counts']}
        seconds = min(value['budget_seconds'], value['elapsed_seconds'])
        value['chains_per_second'] = value['within_budget_counts']['attempted']/seconds
        value['comparison_note'] = ('One prospectively added 120-second CPU20-process pass, '
            'not a repeated AB/BA measurement. Same inputs, fitter and all-start policy.')
        util = [json.loads(line) for line in (process_dir/'utilization.jsonl').read_text().splitlines()]
        value['cpu_mean_busy_percent'] = float(np.average(
            [r['cpu_busy_percent'] for r in util], weights=[r['interval_seconds'] for r in util])) if util else None
        value['paired_with_thread_cpu'] = compare_records(process_dir/'cpu-process.jsonl.gz', p/'cpu-0.jsonl.gz')
        value['paired_with_gpu'] = compare_records(process_dir/'cpu-process.jsonl.gz', p/'gpu-0.jsonl.gz')
        result['supplementary_cpu_process_baseline'] = value
        result['gpu_over_cpu_process_throughput_ratio'] = gpu/value['chains_per_second'] if value['chains_per_second'] else None
    path=p/'longgap.jsonl.gz'
    if path.exists():
        rows=[]
        with gzip.open(path,'rt') as stream:
            for item in map(json.loads,stream):rows.append(item)
        unique={tuple(r['tracklet_indices']) for item in rows for r in item.get('rows',[])}
        result['planetary']['distinct_output_memberships']=len(unique)
        result['planetary']['longest_output_membership']=max((len(k) for k in unique),default=0)
        result['planetary']['roles']={role:{'processed':sum(role in r.get('roles',[]) for r in rows),'with_output':sum(role in r.get('roles',[]) and bool(r.get('rows')) for r in rows)} for role in ['ordinary','distant','retrograde']}
        kinds = defaultdict(set)
        for item in rows:
            for row in item.get('rows', []):
                kinds[row['extension_kind']].add(tuple(row['tracklet_indices']))
        result['planetary']['distinct_memberships_by_output_kind'] = {k: len(v) for k, v in kinds.items()}
        extensions = [row for item in rows for row in item.get('rows', [])
                      if row['extension_kind'] != 'validated_prior_chain']
        result['planetary']['distinct_extension_memberships'] = len({tuple(r['tracklet_indices']) for r in extensions})
        result['planetary']['longest_extension_membership'] = max((len(r['tracklet_indices']) for r in extensions), default=0)
        result['planetary']['largest_extension_arc_days'] = max((r['span_days'] for r in extensions), default=None)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':main()
