"""Read-only comparison of whole-pipeline application timings and outputs."""
import argparse,gzip,json
from pathlib import Path


def index_rows(candidates):
    rows={tuple(sorted(x["tracklet_indices"])):x for x in candidates}
    if len(rows)!=len(candidates):
        raise ValueError("Duplicate catalogue-index membership")
    return rows


def summarize(directory):
    p=Path(directory); report=json.loads((p/'report.json').read_text())
    historical=json.loads((Path(__file__).parent/'benchmark-2026-10-21.json').read_text())
    result={'status':report['status'],'historical':{'elapsed_seconds':historical['resources']['elapsed_seconds'],'timings':historical['timings'],'counts':historical['counts']},'runs':report['phases'],'comparisons':{},'limitations':report['limitations']}
    rows={}
    for phase in report['phases']:
        name=phase['name']; path=p/(name+'-report.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as f: data=json.load(f)
            rows[name]=index_rows(data['candidates'])
            phase['historical_wall_ratio']=historical['resources']['elapsed_seconds']/phase['elapsed_seconds']
            if phase.get('counts',{}).get('triple_seeds_completed'):
                phase['triple_fitting_chains_per_second']=phase['counts']['triple_seeds_completed']/phase['timings']['triple_fitting_seconds']
    for a,b in [('cpu-cold','gpu-cold'),('gpu-cold','gpu-warm')]:
        if a not in rows or b not in rows:continue
        left,right=rows[a],rows[b];common=left.keys()&right.keys()
        ea=json.loads((p/a/'benchmark-extra.json').read_text());eb=json.loads((p/b/'benchmark-extra.json').read_text())
        pa=next(x for x in report['phases'] if x['name']==a);pb=next(x for x in report['phases'] if x['name']==b)
        result['comparisons'][a+'__'+b]={'same_seed_hash':ea['seed_sha256']==eb['seed_sha256'],'shared_retained_memberships':len(common),'only_left':[list(k) for k in sorted(left.keys()-right.keys())],'only_right':[list(k) for k in sorted(right.keys()-left.keys())],'max_best_rms_difference_arcsec':max((abs(left[k]['rms_arcsec']-right[k]['rms_arcsec']) for k in common),default=None),'alternate_mode_count_changes':sum(len(left[k]['modes'])!=len(right[k]['modes']) for k in common),'elapsed_ratio':pa['elapsed_seconds']/pb['elapsed_seconds'],'triple_fit_ratio':pa['timings']['triple_fitting_seconds']/pb['timings']['triple_fitting_seconds']}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);a=parser.parse_args();r=summarize(a.directory);(a.directory/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
