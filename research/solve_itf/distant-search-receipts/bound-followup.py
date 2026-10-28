import os,sys,json,gzip,time,hashlib,multiprocessing as mp,tarfile
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from astropy.time import Time
sys.path[:0]=['/tmp/constellation-ship/src','/tmp/constellation-ship/research/solve_itf']
import distant_followup as shared
from distant_search import write
from magictools.itf.catalogue import Catalogue,sha256
from magictools.itf.fit import read_sites
from magictools.itf.nbody import Ephemeris,JPLGeometry
from magictools.itf.longarc import NightIndex,attach_arcs
P=Path('/tmp/itf-distant-search-20261028');out=P/'bound-followup';out.mkdir(exist_ok=False);base=Path('/workspace/constellation/.cache/solve_itf');source=P/'distant-results/followup/all-candidates.json.gz'
rows=json.load(gzip.open(source,'rt'))['candidates'];old={r['digest'] for r in json.load(open(P/'distant-results/followup/selection.json'))['selected']};selected=[];already=[]
for row in rows:
 mi,m=min(enumerate(row['modes']),key=lambda x:(x[1]['rms_arcsec'],x[0]))
 if m['range_au']<30 or m['eccentricity']>=1:continue
 key=hashlib.sha256(json.dumps([row['tracklet_indices'],m['epoch_mjd_tdb'],m['state_au_au_per_day']],separators=(',',':')).encode()).hexdigest()
 record=dict(digest=key,row=row,mode=m,source_mode=mi,roles=['bound_distant_priority_correction'])
 (already if key in old else selected).append(record)
write(out/'contract.json',{'created_unix':time.time(),'reason':'Original span-first all-conic distant quota can crowd out bound modes. Supplement with ALL best-mode bound>=30AU chains, not just favorable cases; preserve original selection and outcomes.','source_sha256':sha256(source),'selected':selected,'already_searched_digests':[r['digest'] for r in already],'workers':4,'seconds':300,'per_seed_seconds':60,'per_fit_seconds':10,'no_new_vm':True,'incremental_cloud_compute_usd':0,'dynamics_and_selection':'Same original planetary procedure/cuts. Known identities/photometry not used in eligibility. No original outcomes relabelled.','code_sha256':sha256(__file__),'shared_code_sha256':sha256(shared.__file__)})
cat=Catalogue(base/'iteration5-cold-cache');sites=read_sites(base/'ObsCodes.html');endpoints=Time([cat.tracks['t'].min()-66,cat.tracks['t'].max()+2],format='mjd',scale='utc').tdb.mjd
ep=Ephemeris(base/'ephemerides/de440s.bsp',*endpoints);idx=NightIndex(cat,sites,ep);attach_arcs(idx,[r['tracklet_indices'] for r in rows],cat)
args=SimpleNamespace(ephemeris=base/'ephemerides/de440s.bsp',backend='compiled',output=out/'private',sigma_arcsec=.3,seed_seconds=60.,max_seed_seconds=60.,fit_seconds=10.,fast_fit=False,fit_cascade=False,staged=True,bridge_first=True,iterations=2,branches=4,min_gap_days=0.,max_gap_days=0.,max_cone_deg=2.,max_candidates=16,retrieval='position',position_reserve=0,uncertainty_samples=12,direct_arcs=True,swap_limit=0,fast_retrieval=False,no_work_reuse=False)
args.output.mkdir();shared.CTX=(JPLGeometry(cat,sites,ep),idx,ep,args);started=time.monotonic();deadline=started+300;summaries=[]
with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork'),initializer=shared.initialize) as pool,gzip.open(out/'recovery.jsonl.gz','wt',compresslevel=1) as f:
 for future in as_completed([pool.submit(shared.one,(i,r,deadline)) for i,r in enumerate(selected)]):
  r=future.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();summaries.append({k:v for k,v in r.items() if k!='rows'}|{'retained_rows':len(r['rows'])});write(out/'progress.json',dict(completed=len(summaries),selected=len(selected),results=summaries,seconds=time.monotonic()-started))
with tarfile.open(out/'recovery-evidence.tar.gz','w:gz',compresslevel=1) as tar:tar.add(out/'private',arcname='private')
write(out/'summary.json',dict(status='bounded_followup_complete',completed=len(summaries),selected=len(selected),already_done=len(already),results=summaries,seconds=time.monotonic()-started,incremental_cloud_compute_usd=0))
print(json.dumps({'completed':len(summaries),'seconds':time.monotonic()-started,'rows':sum(r['retained_rows'] for r in summaries)},indent=2))
