import json,hashlib,pathlib,sys,shutil
import numpy as np
sys.path.insert(0,'/tmp/constellation-ship/src')
from magictools.itf.catalogue import Catalogue,sha256
P=pathlib.Path('/tmp/itf-distant-search-20261028');root=pathlib.Path('/workspace/constellation/.cache/solve_itf')
# Fixed availability-only rule, saved before any fits: first eligible season,
# up to six multi-detection tracklets within 120 days; first later tracklet >=180d.
rule={'selection':'For each of nine existing control identities, order eligible tracklets by UTC time (count>=2,0<span<=0.6d). Select first starting index with >=3 tracklets in next120d, take first up to6. Reserve first eligible tracklet >=180d after last fitted time. No outcomes used.','roles':'Known catalogue identities are development labels; all observation identities replaced before search; no evaluation split opened. Search background first2000 full-catalogue ordinals with t in[60000,60010), independent of labels.','known_names':{'05145':'Pholus','02060':'Chiron','20000':'Varuna','q8219':'2008 KV42 (retrograde TNO)','p4107':'514107 (retrograde co-orbital)','00433':'Eros','B1309':'111309','00004':'Vesta','01685':'Toro'}}
(P/'control-selection-rule.json').write_text(json.dumps(rule,indent=2))
full=Catalogue(root/'iteration5-cold-cache');pick=np.flatnonzero((full.tracks['t']>=60000)&(full.tracks['t']<60010))[:2000]
tracks=[];obs=[];owners=[];reserved=[];selection=[];cursor=0
for i in pick:
 r=full.tracks[i].copy();o=full.obs[r['start']:r['start']+r['count']].copy();r['start']=cursor;cursor+=len(o);tracks.append(r);obs.append(o);owners.append(None)
for base in ['known-controls','additional-controls']:
 c=Catalogue(root/base/'canonical-catalogue')
 for identity in np.unique(c.tracks['id']):
  eligible=np.flatnonzero((c.tracks['id']==identity)&(c.tracks['count']>=2)&(c.tracks['span']>0)&(c.tracks['span']<=.6));eligible=eligible[np.argsort(c.tracks['t'][eligible],kind='stable')];chosen=[]
  for j,k in enumerate(eligible):
   window=eligible[j:][c.tracks['t'][eligible[j:]]<=c.tracks['t'][k]+120]
   if len(window)>=3:chosen=window[:6];break
  key=identity.decode().strip();record={'identity':key,'name':rule['known_names'].get(key,key),'source':str(c.path),'source_manifest':c.manifest,'selected_original_indices':list(map(int,chosen)),'anonymous_memberships':[],'reserved':None}
  for i in chosen:
   r=c.tracks[i].copy();o=c.obs[r['start']:r['start']+r['count']].copy();r['id']=f'CT{len(tracks):08d}'.encode();r['start']=cursor;cursor+=len(o);record['anonymous_memberships'].append(len(tracks));tracks.append(r);obs.append(o);owners.append(key)
  if len(chosen):
   later=eligible[c.tracks['t'][eligible]>=c.tracks['t'][chosen[-1]]+180]
   if len(later):
    i=int(later[0]);r=c.tracks[i];o=c.obs[r['start']:r['start']+r['count']];rr={'identity':key,'source_original_index':i,'site':r['site'].decode(),'observations':[dict(t=float(x['t']),ra_mas=int(x['ra_mas']),dec_mas=int(x['dec_mas'])) for x in o]};reserved.append(rr);record['reserved']=rr
  selection.append(record)
out=P/'control-catalogue';out.mkdir(exist_ok=True)
np.save(out/'tracklets.npy',np.array(tracks,dtype=full.tracks.dtype));np.save(out/'observations.npy',np.concatenate(obs));manifest={'schema':2,'source_sha256':hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),'counts':{'tracklets':len(tracks),'stored_observations':cursor},'control_warning':'Availability-selected labelled development observations mixed with real unrelated background; not detection completeness or blind validation.'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(P/'control-truth.json').write_text(json.dumps({'selection':selection,'owners':owners,'background_count':len(pick)},indent=2));(P/'control-reserved.json').write_text(json.dumps(reserved,indent=2));print(json.dumps({'tracks':len(tracks),'controls':[(r['name'],len(r['anonymous_memberships']),r['reserved'] is not None) for r in selection]}))
