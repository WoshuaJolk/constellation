import json,gzip,hashlib,sys,time,csv
from pathlib import Path
import numpy as np
from astropy.time import Time
sys.path[:0]=['/tmp/constellation-ship/src','/tmp/constellation-ship/research/solve_itf']
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.fit import read_sites
from magictools.itf.nbody import Ephemeris,JPLGeometry,Trajectory
from magictools.itf.kernels import ARCSEC
from distant_search import write
P=Path('/tmp/itf-distant-search-20261028');base=Path('/workspace/constellation/.cache/solve_itf');source=P/'planetary-candidates.json.gz';raw=json.load(gzip.open(source,'rt'))['candidates'];rows={}
for r in raw:
 k=tuple(r['tracklet_indices'])
 if r['range_au']>=30 and r['eccentricity']<1 and (k not in rows or r['rms_arcsec']<rows[k]['rms_arcsec']):rows[k]=r
# Group directly and transitively shared detections as competing/nested families,
# not as proof that every member represents the same physical object.
remaining=set(rows);families=[]
while remaining:
 key=min(remaining);remaining.remove(key);group=[key];seen=set(key)
 while True:
  add=[k for k in sorted(remaining) if seen.intersection(k)]
  if not add:break
  for k in add:remaining.remove(k);group.append(k);seen.update(k)
 families.append(group)
representatives=[min(g,key=lambda k:(-rows[k]['span_days'],-len(k),rows[k]['rms_arcsec'],k)) for g in families]
write(P/'dossier-replay-contract.json',{'created_unix':time.time(),'source_sha256':sha256(source),'selection':'Every connected shared-tracklet family among planetary rows withboundmode>=30AU. Representative longestarc,thenmosttracklets,thenlowestinputRMS. Descriptive grouping,notindependentobjects.','representatives':[list(k) for k in representatives],'states_sha256':[hashlib.sha256(json.dumps([rows[k]['epoch_mjd_tdb'],rows[k]['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest() for k in representatives],'prediction':'Existing compiled JPL apparent path; same global ephemerisgrid,rtol2e-11,maxstep10d,lighttime; no refit or new data','code_sha256':sha256(__file__)})
cat=Catalogue(base/'iteration5-cold-cache');endpoints=Time([cat.tracks['t'].min()-66,cat.tracks['t'].max()+2],format='mjd',scale='utc').tdb.mjd;ep=Ephemeris(base/'ephemerides/de440s.bsp',*endpoints);g=JPLGeometry(cat,read_sites(base/'ObsCodes.html'),ep)
photo=np.load(base/'photometry-v1/photometry.npy',mmap_mode='r');annotation={r['membership_key']:r for r in csv.DictReader(open(P/'planetary-candidates.csv'))};old=json.load(open(P/'frozen-inventory-comparison.json'))['matches'];all_initial=json.load(gzip.open(P/'distant-results/followup/all-candidates.json.gz','rt'))['candidates'];out=[]
for keys,key in zip(families,representatives):
 row=rows[key];g.prepare(key);t,u,observer,labels=g.get(key);state=np.array(row['state_au_au_per_day']);epoch=row['epoch_mjd_tdb'];traj=Trajectory(state,epoch,max(ep.start,min(t.min(),epoch)-.05),min(ep.end,max(t.max(),epoch)+.01),ep,deadline=time.monotonic()+30,backend='compiled');pred,_,_=traj.apparent(t,observer);east=np.column_stack((-u[:,1],u[:,0],np.zeros(len(u))));east/=np.linalg.norm(east,axis=1)[:,None];north=np.cross(u,east);diff=(pred-u)*ARCSEC;ra=np.sum(diff*east,axis=1);dec=np.sum(diff*north,axis=1);err=np.arctan2(np.linalg.norm(np.cross(pred,u),axis=1),np.sum(pred*u,axis=1))*ARCSEC;rms=float(np.sqrt(np.mean(err**2)))
 detection=[dict(track_index=int(i),timestamp_mjd_tdb=float(tt),observed_ra_deg=float(np.degrees(np.arctan2(uu[1],uu[0]))%360),observed_dec_deg=float(np.degrees(np.arcsin(uu[2]))),predicted_ra_deg=float(np.degrees(np.arctan2(pp[1],pp[0]))%360),predicted_dec_deg=float(np.degrees(np.arcsin(pp[2]))),signed_predicted_minus_observed_ra_cosdec_arcsec=float(rr),signed_predicted_minus_observed_dec_arcsec=float(dd),angular_separation_arcsec=float(ee)) for i,tt,uu,pp,rr,dd,ee in zip(labels,t,u,pred,ra,dec,err)]
 original=[]
 for i in key:
  track=cat.tracks[i];lo=track['start'];hi=lo+track['count'];o=cat.obs[lo:hi];ph=photo[lo:hi]
  original.append({'index':i,'id':track['id'].decode().strip(),'station':track['site'].decode(),'observations':[dict(mjd_utc=float(x['t']),ra_deg=int(x['ra_mas'])/3600000,dec_deg=int(x['dec_mas'])/3600000,magnitude=float(v['magnitude']) if np.isfinite(v['magnitude']) else None,band=v['band'].decode(),photometry_conflict=bool(v['conflict'])) for x,v in zip(o,ph)]})
 kstr='-'.join(map(str,key));comp=[{'membership':r['tracklet_indices'],'best_rms_arcsec':r['rms_arcsec'],'modes':len(r['modes'])} for r in all_initial if set(key).intersection(r['tracklet_indices'])]
 out.append({'representative':row,'group_memberships':[list(k) for k in keys],'annotations':annotation[kstr],'prior_frozen_fit_versions':old.get(kstr,[]),'original_observations':original,'frozen_prediction_replay':{'observations':len(t),'stored_observation_count':row['observations'],'supplied_observations':sum(len(o['observations']) for o in original),'angular_rms_arcsec':rms,'saved_rms_arcsec':row['rms_arcsec'],'difference_arcsec':abs(rms-row['rms_arcsec']),'no_refitting':True,'all_detections':detection},'overlapping_initial_candidate_memberships':comp,'unresolved':['No independent image confirmation.','No calibrated association probability.','Finite unknown/known competitor coverage; older known export sensitivity remains.','Size cannot be established from heterogeneous bands and assumed reflectivity.']})
with gzip.open(P/'pursuit-dossiers.json.gz','wt',compresslevel=1) as f:json.dump({'families':out},f,allow_nan=False)
summary={'shared_tracklet_families':len(out),'bound_distant_memberships':len(rows),'representatives':[{'tracklets':x['representative']['tracklets'],'span_days':x['representative']['span_days'],'rms_arcsec':x['representative']['rms_arcsec'],'range_au':x['representative']['range_au'],'observations':x['representative']['observations'],'known_status':x['annotations']['known_status'],'seen_in_frozen_inventory':bool(x['prior_frozen_fit_versions']),'replay_rms_difference_arcsec':x['frozen_prediction_replay']['difference_arcsec'],'detections_accounted':x['frozen_prediction_replay']['observations']==x['frozen_prediction_replay']['stored_observation_count']} for x in out],'meaning':'Families group overlapping alternatives, not independent confirmedobjects.'};write(P/'pursuit-summary.json',summary);print(json.dumps(summary,indent=2))
