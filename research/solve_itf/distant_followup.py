"""Planetary follow-up of fixed distant-priority and secondary candidate lanes."""
import argparse,gzip,hashlib,json,multiprocessing as mp,os,time,tarfile
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import SimpleNamespace
import numpy as np
from astropy.time import Time
from distant_search import write

CTX=None

def lane_tags(mode):
    tags=[];r=mode.get('range_au',0);q=mode.get('perihelion_au',float('inf'));a=mode.get('semimajor_axis_au');e=mode.get('eccentricity',0);inc=mode.get('inclination_deg',0)
    if r>=30:tags.append('distant')
    if e<1 and a is not None and a>=30:tags.append('TNO_mode')
    if e<1 and 5.2<q<30:tags.append('Centaur_mode')
    if q<1.3:tags.append('NEO_mode')
    if inc>=90:tags.append('retrograde')
    if e>1:tags.append('hyperbolic_mode')
    return tags or ['other']

def select(records):
    # One best-residual mode per chain per lane first; alternatives remain saved
    # and can enter another lane. No known-object labels used for priority.
    quotas={'distant':128,'retrograde':32,'hyperbolic_mode':32,'NEO_mode':32,'other':32};selected={};stats={}
    for lane,limit in quotas.items():
        choices=[]
        for row in records:
            modes=[(mi,m) for mi,m in enumerate(row['modes']) if lane in lane_tags(m) or lane=='other']
            if not modes:continue
            mi,m=min(modes,key=lambda x:(x[1]['rms_arcsec'],x[0]))
            key=hashlib.sha256(json.dumps([row['tracklet_indices'],m['epoch_mjd_tdb'],m['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest()
            # Longer measured arcs and more tracklets before residual tie-break.
            priority=(-row.get('span_days',0),-len(row['tracklet_indices']),m['rms_arcsec'],key)
            choices.append((priority,key,row,mi,m))
        stats[lane]={'available':len(choices),'quota':limit}
        for _,key,row,mi,m in sorted(choices,key=lambda x:x[0])[:limit]:
            if key not in selected:selected[key]=dict(digest=key,row=row,mode=m,source_mode=mi,roles=[])
            selected[key]['roles'].append(lane)
    return list(selected.values()),stats

def initialize():
    global CTX
    g,index,ep,args=CTX;args.output=Path(args.output)/str(os.getpid());args.output.mkdir(parents=True,exist_ok=True);os.environ['ITF_NBODY_BACKEND']='compiled'

def one(task):
    from magictools.itf.longarc import recover_seed
    ordinal,r,deadline=task;g,index,ep,args=CTX;before=time.monotonic()
    seed=(0.,-r['row'].get('span_days',0),ordinal,r['source_mode'],r['row'],r['mode'])
    try:
        key,status,counts,rows=recover_seed(seed,g,index,ep,args,deadline)
        return dict(digest=r['digest'],roles=r['roles'],status=status,counts=dict(counts),rows=rows,seconds=time.monotonic()-before)
    except Exception as e:return dict(digest=r['digest'],roles=r['roles'],status='unresolved_exception',error=type(e).__name__+': '+str(e),rows=[],counts={},seconds=time.monotonic()-before)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--policy-description',default='128 distant,32 retrograde,32 hyperbolic,32 NEO,32 general; one minimum-RMS available mode per chain/lane; order descending span,tracklets then RMS/hash. All raw alternatives saved. No identity labels.');a=p.parse_args();a.output.mkdir(exist_ok=True)
    merged={};sources=[]
    for source in [Path('general-candidates.json.gz'),Path('distant-results/production/candidates.json.gz')]:
        with gzip.open(source,'rt') as f:data=json.load(f)
        for row in data['candidates']:
            k=tuple(row['tracklet_indices'])
            if k not in merged:merged[k]=row;row['search_sources']=[str(source)]
            else:
                old=merged[k];old['search_sources'].append(str(source));seen={(m['epoch_mjd_tdb'],tuple(m['state_au_au_per_day'])) for m in old['modes']}
                old['modes'] += [m for m in row['modes'] if (m['epoch_mjd_tdb'],tuple(m['state_au_au_per_day'])) not in seen]
        sources.append(dict(path=str(source),candidates=len(data['candidates']),status=data['status']))
    rows=list(merged.values());selected,stats=select(rows)
    with gzip.open(a.output/'all-candidates.json.gz','wt',compresslevel=1) as f:json.dump(dict(sources=sources,candidates=rows),f,allow_nan=False)
    write(a.output/'selection.json',dict(policy=a.policy_description,lanes=stats,selected=selected))
    from magictools.itf.catalogue import Catalogue
    from magictools.itf.fit import read_sites
    from magictools.itf.nbody import Ephemeris,JPLGeometry
    from magictools.itf.longarc import NightIndex,attach_arcs
    cat=Catalogue('catalogue');sites=read_sites('ObsCodes.html');endpoints=Time([cat.tracks['t'].min()-66,cat.tracks['t'].max()+2],format='mjd',scale='utc').tdb.mjd
    ep=Ephemeris('de440s.bsp',*endpoints);idx=NightIndex(cat,sites,ep);attach_arcs(idx,[r['tracklet_indices'] for r in rows],cat)
    args=SimpleNamespace(ephemeris=Path('de440s.bsp'),backend='compiled',output=a.output/'private',sigma_arcsec=.3,seed_seconds=60.,max_seed_seconds=60.,fit_seconds=10.,fast_fit=False,fit_cascade=False,staged=True,bridge_first=True,iterations=2,branches=4,min_gap_days=0.,max_gap_days=0.,max_cone_deg=2.,max_candidates=16,retrieval='position',position_reserve=0,uncertainty_samples=12,direct_arcs=True,swap_limit=0,fast_retrieval=False,no_work_reuse=False)
    args.output.mkdir(exist_ok=True)
    global CTX
    CTX=(JPLGeometry(cat,sites,ep),idx,ep,args);started=time.monotonic();deadline=started+900;summaries=[]
    with ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork'),initializer=initialize) as pool,gzip.open(a.output/'recovery.jsonl.gz','wt',compresslevel=1) as f:
        futures=[pool.submit(one,(i,r,deadline)) for i,r in enumerate(selected)]
        for future in as_completed(futures):
            r=future.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();summaries.append({k:v for k,v in r.items() if k!='rows'}|{'retained_rows':len(r['rows'])})
            write(a.output/'summary.json',dict(status='running',completed=len(summaries),selected=len(selected),seconds=time.monotonic()-started,index_counts=idx.counts,results=summaries))
    with tarfile.open(a.output/'recovery-evidence.tar.gz','w:gz',compresslevel=1) as tar:tar.add(a.output/'private',arcname='private')
    write(a.output/'summary.json',dict(status='bounded_followup_complete',completed=len(summaries),selected=len(selected),seconds=time.monotonic()-started,index_counts=idx.counts,status_counts=dict(Counter(r['status'] for r in summaries)),results=summaries,limitations=['A timeout or failed local fit is unresolved, not an exclusion of the association.','Finite seed modes, windows, cones, branch and proposal caps; not exhaustive.','Planetary fits and uncertainty-guided retrieval are follow-up evidence, not confirmation or calibrated probabilities.']))

if __name__=='__main__':main()
