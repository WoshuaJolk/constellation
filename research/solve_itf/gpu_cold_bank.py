"""Build the existing fit_chain 24 range/radial starts on frozen real geometry."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

GRID=[(distance,vr) for distance in (.1,.5,1.,2.,4.,10.,40.,150.) for vr in (0.,-.01,.01)]

def expand(bank):
    bounds=np.r_[0,np.cumsum(np.bincount(bank['case_index']))]
    n=len(bank['states']); k=len(GRID)
    counts=np.repeat(np.diff(bounds),k)
    output_bounds=np.r_[0,np.cumsum(counts)]
    result={name:np.empty((len(bank[name])*k,)+bank[name].shape[1:],np.float64)
            for name in ('offsets','observers','observed','east','north')}
    result['case_index']=np.repeat(np.arange(n*k,dtype=np.int64),counts)
    result['states']=np.empty((n*k,6),np.float64);result['sigma']=bank['sigma']
    for c in range(n):
        lo,hi=bounds[c:c+2];t=bank['offsets'][lo:hi];u=bank['observed'][lo:hi];ob=bank['observers'][lo:hi]
        assert len(t)>=6 and np.all(np.diff(t)>=0) and t[-1]>t[0], 'Invalid chain geometry; no silent exclusions'
        mid=len(t)//2;dt=t[-1]-t[0]
        for g,(distance,vr) in enumerate(GRID):
            index=c*k+g;outlo,outhi=output_bounds[index:index+2]
            r=ob[mid]+distance*u[mid]
            v=(ob[-1]-ob[0]+distance*(u[-1]-u[0]))/dt
            result['states'][index]=np.r_[r,v+vr*u[mid]]
            for name in ('observers','observed','east','north'):result[name][outlo:outhi]=bank[name][lo:hi]
            result['offsets'][outlo:outhi]=t-t[mid]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=dict(np.load(a.input,allow_pickle=False));out=expand(b);np.savez_compressed(a.output,**out)
    receipt={'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'unique_parent_chains':len(b['states']),'starts':len(out['states']),'unique_source_observation_instances':len(b['offsets']),'expanded_observation_instances':len(out['offsets']),'grid_distance_au_radial_au_day':GRID,'ordering':'parent source ordinal, then original fit_chain range/radial grid order','epoch':'Each parent observation midpoint; offsets shifted, no new observation/observer calculations','excluded_starts':0,'scope':'Actual 24 range/radial grid only; no Gauss, extra saved, or broad_starts hypotheses. Parent chains remain previously retained sample.'}
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
