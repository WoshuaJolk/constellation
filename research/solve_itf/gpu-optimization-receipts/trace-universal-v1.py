from pathlib import Path
import inspect,json,math
import numpy as np
from numba import njit
from magictools.itf import derivatives as d
p=Path('/tmp/itf-gpu-reuse-20261028');old=Path('/tmp/itf-gpu-cold-grid-20261028')
b=dict(np.load(old/'bank.npz'));g=dict(np.load(old/'fit-results/gpu-reference.npz'));cases=[x['start'] for x in json.loads((old/'final-cpu-audit.json').read_text())['cases']]
s=inspect.getsource(d.universal_scalars.py_func);s=s[s.index('def universal_scalars'):].replace('def universal_scalars(','def trace_universal(')
lines=[];reasons={};n=0
for lineno,line in enumerate(s.splitlines()):
 if 'return invalid' in line:
  n+=1;reasons[n]={'source_line_relative':lineno+1,'context':s.splitlines()[max(0,lineno-2):lineno+1]};line=line.replace('return invalid',f'return invalid+({n},)')
 elif line.strip().startswith('return x,y,zp,'):line+=',0'
 lines.append(line)
ns=dict(d.universal_scalars.py_func.__globals__);exec('\n'.join(lines),ns);fn=njit(ns['trace_universal'])
bounds=np.r_[0,np.cumsum(np.bincount(b['case_index']))];out=[];parity=[]
for c in cases:
 for i in range(bounds[c],bounds[c+1]):
  emitted=b['offsets'][i]
  for lt in range(20):
   values=d.universal_scalars(g['states'][c],emitted);v=fn(g['states'][c],emitted)
   same=np.array_equal(np.array(values),np.array(v[:14]),equal_nan=True);parity.append(same)
   if not same:raise RuntimeError('Instrumented propagation changed outputs')
   delta=np.array(values[:3])-b['observers'][i];distance=math.sqrt(float(delta@delta))
   if not math.isfinite(distance) or distance<1e-8:
    out.append(dict(start=c,observation_index=int(i),light_time_iteration=lt,emitted_days=float(emitted),reason=int(v[14])));break
   slope=1+float(delta@np.array(values[3:6]))/(d.C*distance)
   updated=emitted-(emitted-b['offsets'][i]+distance/d.C)/slope if math.isfinite(slope) and slope>1e-8 else b['offsets'][i]-distance/d.C
   if abs(updated-emitted)<1e-11:break
   emitted=updated
  if out and out[-1]['start']==c:break
report=dict(scope='Trace prior 28 CPU-invalid final GPU states, no refits or changed tolerances. Local derivative output parity checked per traced invocation.',cases=out,reasons=reasons,exact_trace_parity_calls=len(parity),label='Diagnostic failure mechanism in CPU implementation, not proof CPU or GPU is physically correct')
(p/'universal-failure-trace.json').write_text(json.dumps(report,indent=2));print(json.dumps({'traced':len(out),'reasons':{str(k):sum(x['reason']==k for x in out) for k in reasons},'calls':len(parity)}))
