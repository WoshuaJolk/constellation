"""Read-only summary of the fixed host-batching experiment."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from gpu_deep_summary import compare_records


def load(path):
    with gzip.open(path,'rt') as stream:
        return {r['source_position']:r for r in map(json.loads,stream) if r['within_budget']}


def summarize(directory):
    p=Path(directory);report=json.loads((p/'report.json').read_text());result=dict(status=report['status'],setup=report.get('setup',{}),phases={},comparisons={})
    util=defaultdict(list)
    for line in (p/'utilization.jsonl').read_text().splitlines():
        row=json.loads(line);util[row['phase']].append(row)
    for name,phase in report['phases'].items():
        row=dict(phase);row['chains_per_second']=phase['within_budget_counts']['attempted']/min(phase['elapsed_seconds'],phase['budget_seconds']);samples=util[name]
        if samples:
            row['cpu_busy_percent']=float(np.average([r['cpu_busy_percent'] for r in samples],weights=[r['interval_seconds'] for r in samples]));row['gpu_util_percent']=float(np.mean([r.get('gpu_util_percent',0.) for r in samples]));row['gpu_peak_memory_mib']=max(r.get('gpu_memory_used_mib',0) for r in samples)
        receipts_path=p/(name+'-batches.json')
        if receipts_path.exists() and name!='process_cpu':
            receipts=[r['gpu'] for r in json.loads(receipts_path.read_text()) if r['within_budget']]
            row['cumulative_producer_planning_seconds']=sum(r['planning_seconds'] for r in receipts)
            row['gpu_solve_seconds']=sum(r['gpu_solve_profile']['wall_seconds'] for r in receipts)
            row['cpu_verify_fallback_seconds']=sum(r['cpu_verify_fallback_seconds'] for r in receipts)
            row['cpu_fallback_starts']=sum(r['cpu_fallback_starts'] for r in receipts)
            row['timing_note']='Producer times can overlap each other and GPU execution; throughput uses only actual phase wall clock.'
        result['phases'][name]=row
    for left,right in [('legacy_gpu','compact_gpu'),('legacy_gpu','prefetch_gpu_0'),('compact_gpu','prefetch_gpu_0'),('process_cpu','prefetch_gpu_0')]:
        if left not in report['phases'] or right not in report['phases']:continue
        paths=[p/(name+'.jsonl.gz') for name in (left,right)]
        if not all(path.exists() for path in paths):continue
        comparison=compare_records(*paths);a,b=[load(path) for path in paths];common=a.keys()&b.keys()
        comparison['full_retained_row_differences']=[k for k in sorted(common) if a[k]['row'] is not None and b[k]['row'] is not None and a[k]['row']!=b[k]['row']]
        result['comparisons'][left+'__'+right]=comparison
    prefetch=[r for k,r in report['phases'].items() if k.startswith('prefetch_gpu')]
    if prefetch:
        seconds=sum(min(r['elapsed_seconds'],r['budget_seconds']) for r in prefetch);rate=sum(r['within_budget_counts']['attempted'] for r in prefetch)/seconds
        result['prefetch_chains_per_second']=rate;result['versus_previous_process_baseline']=rate/622.9333333333333
        if 'process_cpu' in result['phases']:result['versus_current_process_cpu']=rate/result['phases']['process_cpu']['chains_per_second']
        if 'legacy_gpu' in result['phases']:result['versus_same_worker_legacy_gpu']=rate/result['phases']['legacy_gpu']['chains_per_second']
        result['target_100x_reached_vs_previous_process_cpu']=rate>=62293.33333333333
    all_positions=set();evaluations=0;starts=0
    for name,phase in report['phases'].items():
        if (p/(name+'.jsonl.gz')).exists():all_positions.update(load(p/(name+'.jsonl.gz')))
        evaluations+=phase['within_budget_counts']['attempted'];starts+=phase['within_budget_counts']['starts']
    result.update(distinct_seed_positions=len(all_positions),timed_chain_evaluations=evaluations,timed_optimizer_starts=starts)
    result['limitations']=['Development performance experiment, not truth/recall evidence.','Full catalogue and existing7Mqueue reused; no fresh graph construction or planetary retrieval timing.','Original numerical/scientific cuts unchanged; CPU/GPU alternate-mode discrepancies remain separately reported.','Periodic utilization samples can straddle phase boundaries.','No new production linking or deployment.']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);a=parser.parse_args();r=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':main()
