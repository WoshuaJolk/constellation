"""Bounded nonlinear proposal exploration; final joint fitting remains mandatory."""
from collections import Counter
import numpy as np
from .fit import check_deadline


def retrieve(index,base,ephemeris,args,minimum,maximum,deadline):
    settings=dict(minimum_gap=minimum,maximum_gap=maximum,max_cone_deg=args.max_cone_deg,
        max_candidates=args.max_candidates,retrieval=getattr(args,'retrieval','position'),
        position_reserve=getattr(args,'position_reserve',0),deadline=deadline)
    central,counts=index.proposals(base,ephemeris,**settings)
    parent_trajectory=index.last_trajectory
    arc_candidates=list(getattr(index,'arc_candidates',central))
    hints={i:(base,parent_trajectory) for i in set(central)|set(arc_candidates)}
    sample_count=getattr(args,'uncertainty_samples',0)
    if not sample_count or not counts.get('uncertainty_gated_nights'):
        return central,counts,hints,arc_candidates
    counts=Counter(counts);lists=[central];arc_lists=[arc_candidates]
    factor=np.asarray(base['covariance_factor']);u,s,_=np.linalg.svd(factor)
    center=np.asarray(base['state_au_au_per_day'])
    # Axis probes discover nonlinear/disconnected retrieval regions without claiming
    # a complete uncertainty volume or posterior quadrature. Conditional cones are
    # proposal heuristics; they never replace the all-observation likelihood.
    probes=[(axis,sign,radius) for radius in (2.5,1.) for axis in range(6) for sign in (-1,1)]
    for axis,sign,radius in probes[:sample_count]:
        check_deadline(deadline)
        trial=dict(base,state_au_au_per_day=(center+sign*radius*s[axis]*u[:,axis]).tolist(),
                   covariance_factor=(factor/3.).tolist())
        counts['uncertainty_probes_attempted']+=1
        try:
            proposals,extra=index.proposals(trial,ephemeris,**settings)
            arcs=list(getattr(index,'arc_candidates',proposals))
            trajectory=index.last_trajectory
            for i in set(proposals)|set(arcs):hints.setdefault(i,(trial,trajectory))
            lists.append(proposals);arc_lists.append(arcs)
            counts['uncertainty_probe_raw_proposals']+=extra.get('raw_proposals',0)
            counts['uncertainty_probe_gated_nights']+=extra.get('uncertainty_gated_nights',0)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):
            counts['uncertainty_probe_numerical_failures']+=1
    def interleave(groups,limit):
        result=[];seen=set()
        for k in range(max(map(len,groups),default=0)):
            for group in groups:
                if k<len(group) and group[k] not in seen:
                    seen.add(group[k]);result.append(group[k])
                    if len(result)==limit:return result
        return result
    proposals=interleave(lists,args.max_candidates)
    arcs=interleave(arc_lists,max(args.max_candidates*4,1))
    counts['uncertainty_proposals_retrieved']=len(set(proposals)-set(central))
    counts['uncertainty_proposals_deferred']=len(set().union(*(set(x) for x in lists)))-len(proposals)
    index.last_trajectory=parent_trajectory
    return proposals,dict(counts),hints,arcs
