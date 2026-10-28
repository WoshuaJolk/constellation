"""Bounded distant-priority development search. Keeps every retained conic.

Labels never enter graph construction or fitting. The known control experiment
is availability-selected development, not a survey completeness estimate.
"""
import argparse,gzip,hashlib,json,multiprocessing as mp,os,time,tarfile,subprocess,sys
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.search import SearchConfig,make_seeds
from magictools.itf.fit import BudgetExpired,read_sites
from magictools.itf.broad import approximate_observers,project,unique_edges,best_edges
from magictools.itf.kernels import ARCSEC

DISTANCES=(10.,20.,30.,40.,60.,100.,150.,300.,1000.)

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,allow_nan=False,indent=2));tmp.replace(path)

def distant_eligible(tracks):
    return ((tracks['count']>=2)&(tracks['span']>0)&(tracks['span']<=.6)&
            (np.linalg.norm(tracks['v'],axis=1)*ARCSEC<=600))

def scan_partition(arguments):
    directory,part,parts,seconds=arguments;directory=Path(directory)
    arrays=[np.load(directory/(name+'.npy'),mmap_mode='r') for name in ('tracks','positions','velocities','days','earth','earth_velocity')]
    tracks,positions,velocities,days,earth,ev=arrays;clock=time.monotonic();deadline=clock+seconds
    ids=np.flatnonzero(distant_eligible(tracks)&np.isfinite(positions).all(axis=1));day=np.floor(tracks['t'][ids]).astype(int);order=np.argsort(day,kind='stable');ids,day=ids[order],day[order]
    nights,starts=np.unique(day,return_index=True);groups={int(d):ids[a:b] for d,a,b in zip(nights,starts,np.r_[starts[1:],len(ids)])};ep=CubicHermiteSpline(days,earth,ev)
    edges=np.empty((0,3));pending=[];size=0;counts=Counter();complete=True;cap=150000
    for ordinal,(day,a) in enumerate(groups.items()):
        if ordinal%parts!=part:continue
        if time.monotonic()>=deadline:complete=False;break
        for distance in DISTANCES:
            for gap in range(1,121):
                if day+gap not in groups:continue
                if time.monotonic()>=deadline:complete=False;break
                b=groups[day+gap];ref=day+.5+gap/2;earth_ref=ep(ref)
                left,ga=project(tracks[a],positions[a],velocities[a],ref,earth_ref,distance,0.)
                right,gb=project(tracks[b],positions[b],velocities[b],ref,earth_ref,distance,0.)
                aa,bb=a[ga],b[gb]
                if not len(aa) or not len(bb):continue
                # Fixed proposal aperture, not a fitted-residual cut.
                gate=(30.+.1*gap*gap)/ARCSEC;k=min(17,len(bb))
                ds,j=cKDTree(right[gb]).query(left[ga],k=list(range(1,k+1)),distance_upper_bound=gate,workers=1)
                counts['queries']+=len(aa);counts['day_distance_pairs']+=1
                if k>16:counts['queries_at_cap']+=int(np.isfinite(ds[:,-1]).sum());ds,j=ds[:,:-1],j[:,:-1]
                ii,jj=np.where(np.isfinite(ds))
                if len(ii):
                    ai,bi=aa[ii],bb[j[ii,jj]];dt=tracks['t'][bi]-tracks['t'][ai]
                    good=(dt>=.3)&(dt<=120)&(tracks['id'][ai]!=tracks['id'][bi]);ai,bi,dt=ai[good],bi[good],dt[good]
                    noise=.3*(np.sqrt(12/tracks['count'][ai])/tracks['span'][ai]+np.sqrt(12/tracks['count'][bi])/tracks['span'][bi]);scale=30.+.5*noise*dt
                    score=.5*(ds[ii[good],jj[good]]*ARCSEC/scale)**2+2*np.log(scale/30)+np.log(len(DISTANCES))
                    pending.append(np.column_stack((ai,bi,score)));size+=len(ai);counts['proposed_edges']+=len(ai)
                if size>cap:
                    combined=unique_edges(np.concatenate([edges]+pending));counts['edge_cap_discarded']+=max(0,len(combined)-cap);edges=best_edges(combined,cap);pending=[];size=0
            if not complete:break
        counts['source_nights_completed']+=int(complete)
        if not complete:break
    combined=unique_edges(np.concatenate([edges]+pending));counts['edge_cap_discarded']+=max(0,len(combined)-cap);edges=best_edges(combined,cap)
    path=directory/f'part-{part}.npy';np.save(path,edges)
    receipt=dict(part=part,complete=complete,counts=dict(counts),edges=len(edges),seconds=time.monotonic()-clock,sha256=sha256(path))
    write(directory/f'part-{part}.json',receipt);return receipt

def graph(cat,sites,out,workers,seconds):
    out.mkdir(exist_ok=True);started=time.monotonic();geom=approximate_observers(cat.tracks,sites)
    for name,a in zip(('tracks','positions','velocities','days','earth','earth_velocity'),(cat.tracks,*geom)):np.save(out/(name+'.npy'),a)
    del geom
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        receipts=list(pool.map(scan_partition,[(str(out),i,workers,seconds) for i in range(workers)]))
    edges=unique_edges(np.concatenate([np.load(out/f'part-{i}.npy') for i in range(workers)]));before=len(edges);edges=best_edges(edges,2000000)
    counts=Counter()
    for r in receipts:counts.update(r['counts'])
    counts.update(tracklets_total=len(cat.tracks),distant_rate_eligible=int(distant_eligible(cat.tracks).sum()),merge_cap_discarded=before-len(edges),edges_retained=len(edges))
    np.save(out/'edges.npy',edges);record=dict(complete=all(r['complete'] for r in receipts),counts=dict(counts),partitions=receipts,seconds=time.monotonic()-started,sha256=sha256(out/'edges.npy'))
    write(out/'summary.json',record)
    # These proposal arrays are regenerable and are not scientific fit evidence.
    for name in ('tracks','positions','velocities','days','earth','earth_velocity'):(out/(name+'.npy')).unlink()
    return edges,record

def solve_graph(g,edges,out,max_seeds,seconds,*,scan_complete=False):
    from magictools.itf import search
    from gpu_stream_adapter_v2 import StreamGPUBatchV2
    cfg=SearchConfig(workers=20,max_gap_days=120,max_edges=max(1,len(edges)),max_seeds=max_seeds,seed_neighbors=0,broad_search=True,seconds=seconds)
    counts={};keys,adj=make_seeds(edges,g.catalogue.tracks,cfg,counts,time.monotonic()+120)
    np.save(out/'seeds.npy',np.asarray(keys,dtype=np.int64).reshape(-1,3));write(out/'seed-freeze.json',dict(counts=counts,sha256=sha256(out/'seeds.npy'),config=asdict(cfg)))
    # Use this frozen graph and ordinary extension machinery; no second graph scan.
    def frozen_edges(tracks,c,deadline,progress,counters):
        counters.update(edges_retained=len(edges),full_catalogue_scan_complete=scan_complete)
        return edges
    def frozen_seeds(e,tracks,c,counters,deadline):
        counters.update(counts)
        return keys,adj
    search.candidate_edges=frozen_edges
    search.make_seeds=frozen_seeds
    import magictools.itf.broad as broad
    broad.augment_edges=lambda original,*a,**kw:original
    with StreamGPUBatchV2() as adapter:
        result=search.solve(g.catalogue,g.sites,cfg,geometry=g,batch_fitter=adapter)
        write(out/'batch-profile.json',dict(receipts=adapter.receipts,pipeline=adapter.pipeline))
    with gzip.open(out/'candidates.json.gz','wt',compresslevel=1) as f:json.dump(result,f,allow_nan=False)
    write(out/'fit-summary.json',{k:v for k,v in result.items() if k!='candidates'}|{'candidates':len(result['candidates'])})
    return result

def controls(out):
    from magictools.itf.campaign_geometry import prepare_atlas
    from magictools.itf.campaign_geometry_fast import prepare_observation_ids,FastAtlasGeometry
    from magictools.itf.search import candidate_edges
    cat=Catalogue('control-catalogue');sites=read_sites('ObsCodes.html');out.mkdir(exist_ok=True)
    prepare_atlas(cat.path,'ObsCodes.html',out/'atlas');prepare_observation_ids(cat.path,out/'identities')
    curved,record=graph(cat,sites,out/'graph',4,180)
    cfg=SearchConfig(workers=4,max_gap_days=60,max_edges=300000,max_seeds=50000)
    linear_counts={};linear=candidate_edges(cat.tracks,cfg,time.monotonic()+120,print,linear_counts)
    edges=unique_edges(np.concatenate((curved,linear)))
    g=FastAtlasGeometry(cat.path,'ObsCodes.html',out/'atlas',out/'identities');result=solve_graph(g,edges,out,50000,300,scan_complete=record['complete'] and linear_counts.get('full_catalogue_scan_complete',False))
    # Labels are opened only after the search outputs have been saved.
    truth=json.loads(Path('control-truth.json').read_text());owners=truth['owners'];grading=[]
    for case in truth['selection']:
        pure=[];mixed=[]
        for row in result['candidates']:
            labels=[owners[i] for i in row['tracklet_indices']]
            if case['identity'] in labels:(pure if all(x==case['identity'] for x in labels) else mixed).append(row)
        grading.append(dict(identity=case['identity'],name=case['name'],input_tracklets=len(case['anonymous_memberships']),pure_chains=len(pure),mixed_chains=len(mixed),longest_pure=max([len(r['tracklet_indices']) for r in pure],default=0),longest_span_days=max([r.get('span_days',0) for r in pure],default=0)))
    write(out/'recovery.json',dict(cases=grading,search_status=result['status'],graph_complete=record['complete'],limitations=['Real catalogue labels are development assumptions.','Known seasons are availability-selected; not detection completeness.','The reserved later tracklets did not enter search/fits.']))
    # Frozen nominal prediction diagnostic, every pure retained mode, no best-after-outcome choice.
    from magictools.itf.fit import Geometry
    from magictools.itf.catalogue import OBS_DTYPE,TRACK_DTYPE
    from magictools.itf.kernels import directions
    later=[]
    for case in truth['selection']:
        test=case['reserved']
        if test is None:later.append(dict(identity=case['identity'],status='NO_TEST_DATA'));continue
        o=np.array([(v['t'],v['ra_mas'],v['dec_mas']) for v in test['observations']],dtype=OBS_DTYPE);r=np.zeros(1,TRACK_DTYPE);r['start']=0;r['count']=len(o);r['site']=test['site'].encode()
        fake=SimpleNamespace(tracks=r,obs=o);gg=Geometry(fake,sites);gg.prepare([0]);_,tt,uu,oo=gg.prepared[0]
        for row in result['candidates']:
            if not all(owners[i]==case['identity'] for i in row['tracklet_indices']):continue
            for mi,m in enumerate(row['modes']):
                pred=directions(np.array(m['state_au_au_per_day']),tt-m['epoch_mjd_tdb'],oo);err=np.arctan2(np.linalg.norm(np.cross(pred,uu),axis=1),np.sum(pred*uu,axis=1))*ARCSEC
                later.append(dict(identity=case['identity'],membership=row['tracklet_indices'],mode=mi,rms_arcsec=float(np.sqrt(np.mean(err**2))) if np.isfinite(err).all() else None,errors_arcsec=[float(x) if np.isfinite(x) else None for x in err],gap_days=float(tt.min()-max(g.get(row['tracklet_indices'])[0])),status='predicted' if np.isfinite(err).all() else 'numerical_failure'))
    write(out/'later-predictions.json',dict(results=later,interpretation='Nominal two-body prediction only; no uncertainty coverage or confirmation. All pure modes preserved.'))

def production(out):
    from gpu_scaling_benchmark import geometry
    out.mkdir(exist_ok=True);g=geometry();edges,receipt=graph(g.catalogue,g.sites,out/'graph',20,600)
    result=solve_graph(g,edges,out,1000000,1500,scan_complete=receipt['complete'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['controls','production']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.phase=='controls':controls(a.output)
    else:production(a.output)

if __name__=='__main__':main()
