"""Read-only output, timing and resource comparisons for integrated GPU search."""
import argparse,gzip,json
from pathlib import Path


def summarize(directory):
    p=Path(directory);report=json.loads((p/'report.json').read_text())
    result={'status':report['status'],'phases':report['phases'],'comparisons':{},'input_verification_seconds':report.get('input_verification_seconds')}
    rows={};extras={};phases={x['name']:x for x in report['phases']}
    for name,phase in phases.items():
        file=p/(name+'-report.json.gz')
        if not file.exists():continue
        with gzip.open(file,'rt') as f:data=json.load(f)
        rows[name]={tuple(sorted(x['tracklet_indices'])):x for x in data['candidates']}
        assert len(rows[name])==len(data['candidates']),'Exact memberships must be unique'
        extra=p/name/'extra.json'
        if extra.exists():
            e=json.loads(extra.read_text());extras[name]=e
            phase['seed_sha256']=e['seed_sha256'];phase['pipeline']=e.get('pipeline');phase['resident_geometry']=e.get('resident_geometry')
            receipts=e['batch_receipts']
            if receipts:
                phase['batch_totals']={key:sum(r.get(key,0) for r in receipts) for key in ['chains','starts','planning_seconds','gpu_setup_seconds','cpu_verify_fallback_seconds','cpu_fallback_starts','selection_seconds','execution_seconds']}
                phase['batch_totals']['gpu_solve_seconds']=sum(r['gpu_solve_profile']['wall_seconds'] for r in receipts)
                phase['batch_totals']['note']='Preparation overlaps execution; phase elapsed, not sum, is headline wall time.'
    pairs=[('cpu4-10k','cpu20-10k'),('cpu20-10k','compact-10k'),('compact-10k','prefetch-10k'),('prefetch-10k','indexed-10k'),('cpu20-10k','prefetch-10k'),('cpu20-large','prefetch-large'),('prefetch-large','indexed-large'),('cpu20-large','indexed-large')]
    for a,b in pairs:
        if a not in rows or b not in rows or a not in extras or b not in extras:continue
        left,right=rows[a],rows[b];common=left.keys()&right.keys()
        result['comparisons'][a+'__'+b]={'same_seed_hash':extras[a]['seed_sha256']==extras[b]['seed_sha256'],'shared_memberships':len(common),'only_left':[list(k) for k in sorted(left.keys()-right.keys())],'only_right':[list(k) for k in sorted(right.keys()-left.keys())],'max_best_rms_difference_arcsec':max((abs(left[k]['rms_arcsec']-right[k]['rms_arcsec']) for k in common),default=None),'alternate_mode_count_changes':sum(len(left[k]['modes'])!=len(right[k]['modes']) for k in common),'elapsed_ratio':phases[a]['elapsed_seconds']/phases[b]['elapsed_seconds'],'triple_fitting_ratio':phases[a]['timings']['triple_fitting_seconds']/phases[b]['timings']['triple_fitting_seconds']}
    if (p/'kernel-profile.json').exists():result['kernel_profile']=json.loads((p/'kernel-profile.json').read_text())
    result['limitations']=['Single fixed-order passes; no significance or purity claims.','Same cached input/atlas reused; no claim of cold ingestion included.','Large seed-neighbor policy is broader than10K, but identical between matched backends.','Fitted starts, scientific cuts and observations unchanged across each matched cohort.','GPU initialization, CPU prefetch startup/drain, scans, fitting, growth and output are included in elapsed.','A cap exceeding available seeds does not imply that many seeds were fitted.']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);a=parser.parse_args();r=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
