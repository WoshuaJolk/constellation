"""Freeze real catalogue inputs and previously opened long-gap controls."""
import argparse, gzip, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from magictools.itf.catalogue import Catalogue, OBS_DTYPE, TRACK_DTYPE, sha256
from magictools.itf.fit import Geometry, read_sites
from magictools.itf.kernels import directions, summaries, ARCSEC
from magictools.itf.broad import approximate_observers
from distant_search import distant_eligible
from distant_epoch_graph import DISTANCES, RADIAL_FRACTIONS


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--sites',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);stamp=time.perf_counter()
    design=dict(version=1,
      projection='All eligible rows within365d of floor(median eligible UTC/180)*180; if >131072, evenly spaced original ordinals. Distinct prefix sizes1024,16384,65536,full. All48 distance/radial hypotheses; no fit outcomes.',
      retrieval='All finite local count>=2 tracklets, moved linearly to own UTC-night center; no magnitude or rate cut. 262144 evenly spaced source ordinals. Half query own night, half first available night >=365d later, terminal wrapped to first night. Radii1,60,360,3600arcsec cycle. These are geometry stress queries, not propagated orbits.',
      science='All9 previously opened controls. Previously selected representative membership, every saved mode; first previously reserved later tracklet only. Anonymous query bank,120arcsec nominal center screening; refine every hit at actual detection timestamps/stations,30arcsec maximum per detection. Fixed engineering diagnostics, not a calibrated rule.',
      restrictions='No refitting, no new source acquisition, no production search, no outcome-dependent replacement. Full ITF holdings plus explicitly inserted reserved controls; neither sky census nor untouched validation.',
      repeats=3,cpu_threads=20,projection_parity_arcsec=1e-5,query_distance_tolerance=1e-15,
      budget=dict(maximum_incremental_compute_usd=5,maximum_worker_lifetime_seconds=3600,workers=1))
    write(a.output/'design.json',design)
    cat=Catalogue(a.catalogue);tr=cat.tracks;sites=read_sites(a.sites)
    eligible=np.flatnonzero(distant_eligible(tr));ref=float(np.floor(np.median(tr['t'][eligible])/180)*180)
    ids=eligible[np.abs(tr['t'][eligible]-ref)<=365]
    if len(ids)>131072:ids=ids[np.linspace(0,len(ids)-1,131072,dtype=int)]
    selected=tr[ids].copy();pos,vel,days,earth,ev=approximate_observers(selected,sites)
    good=np.isfinite(pos).all(axis=1)&np.isfinite(vel).all(axis=1)
    np.savez_compressed(a.output/'projection.npz',t=selected['t'][good],u=selected['u'][good],du=selected['v'][good],
        observers=pos[good],velocities=vel[good],ref=ref,earth=CubicHermiteSpline(days,earth,ev)(ref),
        hypotheses=np.array([(d,r) for d in DISTANCES for r in RADIAL_FRACTIONS]),catalogue_indices=ids[good])
    # Append reserved controls only if no equivalent whole tracklet already exists.
    truth=json.loads((a.controls/'control-truth.json').read_text());cases=truth['selection']
    alltracks=tr.copy();extra_obs=[];reserved_ids={};insertion=[];nextobs=len(cat.obs)
    for ci,c in enumerate(cases):
        reserved=c['reserved']
        if reserved is None:continue
        obs=np.array([(o['t'],o['ra_mas'],o['dec_mas']) for o in reserved['observations']],dtype=OBS_DTYPE)
        possible=np.flatnonzero((tr['site']==reserved['site'].encode())&(tr['count']==len(obs))&(np.abs(tr['t']-obs['t'].mean())<1))
        match=None
        for i in possible:
            old=cat.obs[tr[i]['start']:tr[i]['start']+tr[i]['count']]
            if np.all(np.abs(old['t']-obs['t'])<=.001/86400) and np.all(np.abs(old['ra_mas'].astype('i8')-obs['ra_mas'])<=1) and np.all(np.abs(old['dec_mas'].astype('i8')-obs['dec_mas'])<=1):match=int(i);break
        if match is None:
            from magictools.itf.catalogue import unit_vectors
            row=np.zeros(1,TRACK_DTYPE);row['id']=f'anon{ci:05d}'.encode();row['site']=reserved['site'].encode();row['start']=nextobs;row['count']=len(obs)
            stat=summaries(obs['t'],unit_vectors(obs['ra_mas']/3600000,obs['dec_mas']/3600000),np.array([0]),np.array([len(obs)]))[0]
            row['t'],row['u'],row['v'],row['rms'],row['span']=stat[0],stat[1:4],stat[4:7],stat[7],stat[8]
            match=len(alltracks);alltracks=np.concatenate([alltracks,row]);extra_obs.append(obs);nextobs+=len(obs)
            insertion.append(dict(case=ci,tracklet=match,inserted=True))
        else:insertion.append(dict(case=ci,tracklet=match,inserted=False))
        reserved_ids[ci]=match
    ids=np.flatnonzero((alltracks['count']>=2)&np.isfinite(alltracks['u']).all(axis=1)&np.isfinite(alltracks['v']).all(axis=1)&np.isfinite(alltracks['t']))
    day=np.floor(alltracks['t'][ids]).astype(np.int64);order=np.argsort(day,kind='stable');ids,day=ids[order],day[order]
    unique_days,starts=np.unique(day,return_index=True);ends=np.r_[starts[1:],len(ids)].astype('i8');starts=starts.astype('i8')
    points=alltracks['u'][ids]+alltracks['v'][ids]*(day+.5-alltracks['t'][ids])[:,None];points/=np.linalg.norm(points,axis=1)[:,None]
    for lo,hi in zip(starts,ends):
        order=np.argsort(points[lo:hi,0],kind='stable');points[lo:hi]=points[lo:hi][order];ids[lo:hi]=ids[lo:hi][order]
    source=np.linspace(0,len(ids)-1,min(262144,len(ids)),dtype=int)
    query_nights=np.searchsorted(ends,source,side='right').astype('i8')
    later=np.searchsorted(unique_days,unique_days[query_nights]+365);later[later==len(unique_days)]=0
    query_nights[1::2]=later[1::2]
    radii=2*np.sin(np.resize(np.array([1.,60.,360.,3600.]),len(source))/ARCSEC/2)
    np.savez_compressed(a.output/'retrieval.npz',points=points,starts=starts,ends=ends,days=unique_days,tracklet_ids=ids,
        nights=query_nights,predictions=points[source],radii=radii,source_point_indices=source)
    rows=json.load(gzip.open(a.controls/'distant-results/controls/candidates.json.gz','rt'))['candidates']
    recovery=json.loads((a.controls/'distant-results/controls/orbit-recovery.json').read_text())['cases']
    queries=[];grading=[]
    for ci,c in enumerate(cases):
        representative=next(x for x in recovery if x['identity']==c['identity'])
        grade=dict(case=ci,name=c['name'],identity=c['identity'],reserved_tracklet=reserved_ids.get(ci),status=representative['status'])
        grading.append(grade)
        if ci not in reserved_ids or representative['status']!='recovered':continue
        row=next(r for r in rows if r['tracklet_indices']==representative['selected_membership'])
        test=alltracks[reserved_ids[ci]];day=int(np.floor(test['t']));night=int(np.searchsorted(unique_days,day))
        one=np.array([(day+.5,0,0)],dtype=OBS_DTYPE);ft=np.zeros(1,TRACK_DTYPE);ft['count']=1;ft['site']=test['site']
        gg=Geometry(SimpleNamespace(tracks=ft,obs=one),sites);gg.prepare([0]);_,tt,_,oo=gg.prepared[0]
        for mi,m in enumerate(row['modes']):
            prediction=directions(np.array(m['state_au_au_per_day']),tt-m['epoch_mjd_tdb'],oo)[0]
            queries.append(dict(case=ci,mode=mi,night=night,prediction=prediction.tolist(),
                state=m['state_au_au_per_day'],epoch=m['epoch_mjd_tdb'],fit_membership=row['tracklet_indices'],
                gap_days=representative['later']['gap_days'],finite=bool(np.isfinite(prediction).all()),
                state_sha256=hashlib.sha256(np.array([m['epoch_mjd_tdb']]+m['state_au_au_per_day'],dtype='<f8').tobytes()).hexdigest()))
    write(a.output/'science-queries.json',queries);write(a.output/'grading.json',grading)
    science_days={int(np.floor(alltracks[i]['t'])) for i in reserved_ids.values()}
    si=np.flatnonzero(np.isin(np.floor(alltracks['t']).astype('i8'),list(science_days)))
    blocks=[];compact=alltracks[si].copy();start=0;added=np.concatenate(extra_obs) if extra_obs else np.empty(0,OBS_DTYPE)
    for j,i in enumerate(si):
        t=alltracks[i];lo=int(t['start']);hi=lo+int(t['count']);block=cat.obs[lo:hi] if lo<len(cat.obs) else added[lo-len(cat.obs):hi-len(cat.obs)]
        blocks.append(block);compact[j]['start']=start;start+=len(block)
    np.savez_compressed(a.output/'science-observations.npz',indices=si,tracks=compact,observations=np.concatenate(blocks))
    write(a.output/'input-summary.json',dict(catalogue_tracklets=len(tr),retrieval_points=len(points),retrieval_nights=len(starts),
        retrieval_queries=len(source),projection_tracklets=int(good.sum()),projection_unsupported=int((~good).sum()),projection_reference_utc=ref,
        control_insertions=insertion,control_cases=len(cases),control_queries=len(queries),science_pool_tracklets=len(si),
        catalogue_manifest_sha256=sha256(a.catalogue/'manifest.json'),control_truth_sha256=sha256(a.controls/'control-truth.json'),
        control_fit_sha256=sha256(a.controls/'distant-results/controls/candidates.json.gz'),seconds=time.perf_counter()-stamp))
    write(a.output/'manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(a.output.iterdir()) if p.is_file()})


if __name__=='__main__':main()
