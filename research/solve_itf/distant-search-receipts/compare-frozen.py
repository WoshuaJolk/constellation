import gzip,json,hashlib
from pathlib import Path
P=Path('/tmp/itf-distant-search-20261028');base=Path('/workspace/constellation/runs/skyknown-bulk/20261023-import');source=base/'frozen-fit-annotations.jsonl.gz'
metadata=json.load(open(base/'summary.json'));assert metadata['catalogue']['source_sha256']=='f3bb8bd234c1e9438f165f6bcaae1d893a4d6f852ed14bb3426ed634e037b624'
current=json.load(gzip.open(P/'distant-results/followup/all-candidates.json.gz','rt'))['candidates'];planetary=json.load(gzip.open(P/'planetary-candidates.json.gz','rt'))['candidates'];keys={tuple(sorted(r['tracklet_indices'])) for r in current+planetary};priority={tuple(sorted(r['tracklet_indices'])) for r in planetary};matches={};n=0
for line in gzip.open(source,'rt'):
 r=json.loads(line);k=r['known'];members=tuple(sorted({v['track_index'] for v in k['evidence']}|set(k['missing_track_indices'])));assert len(members)==k['total'],r['fit_version'];n+=1
 if members in keys:matches.setdefault('-'.join(map(str,members)),[]).append(r['fit_version'])
receipt={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'historical_fit_versions_checked':n,'current_union_memberships':len(keys),'already_present_memberships':len(matches),'absent_memberships':len(keys)-len(matches),'planetary_unique_memberships':len(priority),'planetary_already_present':sum('-'.join(map(str,k)) in matches for k in priority),'matches':matches,'scope':'Exact track-index membership comparison to all312133 frozen fit versions, retaining coverage-missing indices; absence is not discovery or absence from every external catalogue.'}
(P/'frozen-inventory-comparison.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='matches'},indent=2))
