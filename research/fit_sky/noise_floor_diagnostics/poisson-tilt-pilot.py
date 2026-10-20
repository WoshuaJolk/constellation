import json,time
import numpy as np
from scipy.optimize import minimize
from research.fit_sky.scene import Scene
from research.fit_sky.statistical import BatchModel,root_for_classes
from flint import ctx
start=time.perf_counter();rng=np.random.default_rng(20261102)
m=json.load(open('research/fit_sky/results/broad-inputs.json'))['scene']
with ctx.workprec(100):s=Scene(m)
model=BatchModel(s);n=100000
classes=rng.choice(len(s.types),n,p=np.array([x.get('weight',1) for x in s.types])/sum(x.get('weight',1) for x in s.types))
x=np.empty((n,len(s.data)))
for cls in range(len(s.types)):
 rows=np.flatnonzero(classes==cls);root=root_for_classes(s,[cls]);bounds=np.array(root['ranges']);points=bounds[:,0]+rng.random((len(rows),len(bounds)))*(bounds[:,1]-bounds[:,0])
 for axis,name in enumerate(root['names']):
  if name.startswith('instrument.'):points[:,axis]=bounds[axis].mean()
 for start_row in range(0,len(rows),4096):
  selected=rows[start_row:start_row+4096];x[selected]=model.means(root,points[start_row:start_row+4096])
y=np.array(list(map(float,s.data)));variance=np.array(s.sigmas)**2;lam=m['count_prior']['poisson_mean']
def value(a):
 g=x@a
 if g.max()>500:return 1e250,np.zeros_like(a)
 exp=np.exp(g)
 f=.5*(a*a*variance).sum()-a@y+lam*(exp.mean()-1)
 grad=a*variance-y+lam*(x.T@exp)/n
 return f,grad
fit=minimize(value,np.zeros(len(y)),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12,'gtol':1e-7})
a=fit.x;score=x@a
result={'status':'uncertified_pilot_only','mean_source_samples':n,'seed':20261102,'instrument_nuisance_fixed_only_for_this_diagnostic':True,'optimized':bool(fit.success),'objective':float(fit.fun),'estimated_tilted_mean_count':float(lam*np.exp(score).mean()),'score_range':[float(score.min()),float(score.max())],'dual':a.tolist(),'zero_catalogue_log_likelihood':float(-.5*np.sum(y*y/variance)),'seconds':time.perf_counter()-start,'full_objective_complete':False}
print(json.dumps(result),flush=True)
json.dump(result,open('/tmp/fitsky-poisson-tilt-pilot.json','w'),indent=2)
# Plain complete-prior evidence point estimate: deliberately not a guarantee.
N=200000;sumw=sumsq=0.;maximum=0.
for start_row in range(0,N,4096):
 size=min(4096,N-start_row);counts=rng.poisson(lam,size);groups={}
 for count in np.unique(counts):
  ids=np.flatnonzero(counts==count)
  for row in ids:
   key=tuple(sorted(rng.choice(len(s.types),int(count)).tolist()));groups.setdefault(key,[]).append(row)
 for cls,rows in groups.items():
  root=root_for_classes(s,cls);bounds=np.array(root['ranges']).reshape(-1,2);p=bounds[:,0]+rng.random((len(rows),len(bounds)))*(bounds[:,1]-bounds[:,0]);w=model.likelihood(root,p)
  sumw+=w.sum();sumsq+=w@w;maximum=max(maximum,float(w.max()))
result['complete_prior_point_estimate_not_a_certificate']=float(sumw/N);result['maximum_observed_likelihood']=maximum;result['naive_standard_error_not_a_certificate']=float(np.sqrt(max(0,sumsq/N-(sumw/N)**2)/N));result['seconds']=time.perf_counter()-start
json.dump(result,open('/tmp/fitsky-poisson-tilt-pilot.json','w'),indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='dual'}),flush=True)
