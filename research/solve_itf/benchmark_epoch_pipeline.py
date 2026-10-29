"""One full real epoch window, all48 hypotheses; paired20-worker stage timing."""
import argparse, hashlib, json, multiprocessing as mp, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from magictools.itf.kernels import MU
from magictools.itf.catalogue import sha256
from distant_epoch_graph import projected
from gpu_distant_projection import ProjectionBank
from benchmark_sparse_epoch_join import nightly, shared

DATA=None
META=None
ROOT=None


def initialize(inputs,output):
    global DATA,META,ROOT
    d=np.load(Path(inputs)/'projection.npz');DATA={k:d[k] for k in d.files}
    m=np.load(Path(inputs)/'projection-metadata.npz');META={k:m[k] for k in m.files};ROOT=Path(output)


def ready(_):return True


def task(item):
    name,h,rep=item;stamp=time.perf_counter();d,f=DATA['hypotheses'][h];arrays=[DATA[k] for k in ('t','u','du','observers','velocities')]
    if name=='cpu':p=projected(*arrays,float(DATA['ref']),DATA['earth'],d,f*np.sqrt(2*MU/d))
    else:p=np.load(ROOT/'projected.npy',mmap_mode='r')[h]
    valid=np.isfinite(p).all(axis=1);noise=.3*np.sqrt(12/META['count'])/META['span']
    edges,counts=(nightly if name=='cpu' else shared)(p[valid],DATA['t'][valid],noise[valid],META['identity'][valid])
    # Restore original selected-row indices before cross-backend comparison.
    ids=np.flatnonzero(valid)
    if len(edges):edges[:,:2]=ids[edges[:,:2].astype(int)]
    path=ROOT/f'{name}-{rep}-{h}.npy';np.save(path,edges)
    return dict(hypothesis=h,seconds=time.perf_counter()-stamp,edges=len(edges),valid=int(valid.sum()),
        counts=counts,sha256=sha256(path),path=path.name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
    design=dict(input_sha256=sha256(a.inputs/'projection.npz'),metadata_sha256=sha256(a.inputs/'projection-metadata.npz'),
        scope='Entire frozen50497-tracklet epoch, all48 hypotheses,20CPUworkers. CPU projection+per-night trees versus GPUprojection+shared CPUindex. No fitting; no completeness outside this window.',
        repeats=2,order=['cpu,gpu','gpu,cpu'],workers=20,absolute_score_parity=1e-8,raw_neighbor_cap_per_hypothesis=8000000)
    (a.output/'design.json').write_text(json.dumps(design,indent=2));report=dict(design=design,runs=[])
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    data=np.load(a.inputs/'projection.npz');arrays=[data[k] for k in ('t','u','du','observers','velocities')];h=data['hypotheses'];ref=float(data['ref']);earth=data['earth']
    # Compile CPU propagation before fork; GPU initialized only after workers start.
    stamp=time.perf_counter();projected(*[x[:1] for x in arrays],ref,earth,float(h[0,0]),float(h[0,1]*np.sqrt(2*MU/h[0,0])))
    report['cpu_compile_seconds']=time.perf_counter()-stamp
    with ProcessPoolExecutor(max_workers=20,mp_context=mp.get_context('fork'),initializer=initialize,initargs=(str(a.inputs),str(a.output))) as pool:
        list(pool.map(ready,range(20)))
        bank=ProjectionBank(*arrays);stamp=time.perf_counter();bank.project(ref,earth,h[:1]);report['gpu_warmup_seconds']=time.perf_counter()-stamp;report['gpu_upload_seconds']=bank.upload_seconds
        for rep in range(2):
            row=dict(repeat=rep,backends={});report['runs'].append(row)
            for name in (('cpu','gpu') if rep==0 else ('gpu','cpu')):
                stamp=time.perf_counter();projection=None
                if name=='gpu':
                    p,projection=bank.project(ref,earth,h);np.save(a.output/'projected.npy',p)
                results=list(pool.map(task,[(name,i,rep) for i in range(len(h))]))
                row['backends'][name]=dict(wall_seconds=time.perf_counter()-stamp,tasks=results,projection=projection);save()
            comparisons=[]
            for i in range(len(h)):
                old=np.load(a.output/f'cpu-{rep}-{i}.npy');new=np.load(a.output/f'gpu-{rep}-{i}.npy')
                equal=bool(np.array_equal(old[:,:2],new[:,:2]));score=float(np.max(np.abs(old[:,2]-new[:,2]),initial=0)) if equal else None
                comparisons.append(dict(hypothesis=i,memberships_equal=equal,maximum_score_difference=score,passed=bool(equal and score<=1e-8)))
            row['comparisons']=comparisons;save()
            if not all(x['passed'] for x in comparisons):
                report['status']='parity_failed';save();return
    # Compact returned raw evidence: all per-hypothesis arrays are retained together.
    import tarfile
    with tarfile.open(a.output/'edges-evidence.tar.gz','w:gz',compresslevel=1) as archive:
        for p in sorted(a.output.glob('*.npy')):
            if p.name=='projected.npy':continue
            archive.add(p,arcname=p.name)
    for p in a.output.glob('*.npy'):p.unlink()
    report['evidence_sha256']=sha256(a.output/'edges-evidence.tar.gz');report['status']='completed';save()


if __name__=='__main__':main()
