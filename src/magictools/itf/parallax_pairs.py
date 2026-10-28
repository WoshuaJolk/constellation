"""Supplemental annual-parallax proposals for slow ITF tracklets, not acceptances."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicHermiteSpline
from .broad import project,unique_edges,best_edges
from .campaign_graph import save_array,save_json,csr_graph
from .catalogue import Catalogue,sha256
from .kernels import ARCSEC


def slow_pairs(tracks,positions,velocities,days,earth,earth_velocity,*,rate_limit=180.,
               gaps=(120,180,365,730),slack=7,ranges=(30.,60.,150.,500.,1000.),limit=100000,progress=print):
    valid=(tracks['count']>=3)&(tracks['span']>0)&np.isfinite(positions).all(axis=1)
    valid &= np.linalg.norm(tracks['v'],axis=1)*ARCSEC<=rate_limit
    chosen=np.flatnonzero(valid);dates=np.floor(tracks['t'][chosen]).astype(int)
    groups={int(day):chosen[dates==day] for day in np.unique(dates)}
    ephemeris=CubicHermiteSpline(days,earth,earth_velocity)
    parts=[];count=0;queries=0;capped=0;last=time.monotonic()
    for day,a in groups.items():
        for gap in gaps:
            for delta in range(-slack,slack+1):
                b=groups.get(day+gap+delta)
                if b is None:continue
                ref=(day+day+gap+delta+1)/2
                earth_ref=ephemeris(ref)
                for distance in ranges:
                    left,ga=project(tracks[a],positions[a],velocities[a],ref,earth_ref,distance,0.)
                    right,gb=project(tracks[b],positions[b],velocities[b],ref,earth_ref,distance,0.)
                    aa,bb=a[ga],b[gb]
                    if not len(aa) or not len(bb):continue
                    k=min(9,len(bb));gate=np.deg2rad(.5)
                    mismatch,near=cKDTree(right[gb]).query(left[ga],k=list(range(1,k+1)),distance_upper_bound=gate)
                    queries+=len(aa)
                    if k==9:
                        capped+=int(np.isfinite(mismatch[:,-1]).sum());mismatch=mismatch[:,:8];near=near[:,:8]
                    rows,cols=np.where(np.isfinite(mismatch))
                    if not len(rows):continue
                    ai,bi=aa[rows],bb[near[rows,cols]]
                    keep=tracks['id'][ai]!=tracks['id'][bi]
                    score=(mismatch[rows,cols][keep]/gate)**2
                    parts.append(np.column_stack((ai[keep],bi[keep],score)))
                    count+=int(keep.sum())
                if count>4*limit:
                    parts=[best_edges(unique_edges(np.concatenate(parts)),limit)];count=len(parts[0])
        if time.monotonic()-last>15:
            progress(f'Slow parallax lane: day {day}, {queries:,} queries');last=time.monotonic()
    edges=unique_edges(np.concatenate(parts)) if parts else np.empty((0,3))
    return best_edges(edges,limit),dict(slow_tracklets=len(chosen),queries=queries,neighbor_queries_at_cap=capped,
        retained_edges=min(limit,len(edges)),rate_limit_arcsec_day=rate_limit,gaps_days=list(gaps),
        slack_days=slack,ranges_au=list(ranges),radial_rate_au_day=0.,gate_deg=.5,
        limitations='Supplemental approximate quadratic proposal lane; finite range/time/rate gates, not exhaustive')


def run(args):
    c=Catalogue(args.cache);directory=args.output/'graph'
    if not (directory/'csr.json').exists():raise ValueError('Complete base graph first')
    if (directory/'extended-csr.json').exists():
        saved=__import__('json').loads((directory/'parallax.json').read_text())
        # Completed immutable proposal products are reusable after seed commit.
        print('Supplemental parallax graph already committed',flush=True)
        return
    if (args.output/'seeds.json').exists():raise ValueError('Cannot change graph after committing seeds')
    marker=directory/'parallax.json'
    if not marker.exists():
        arrays=[np.load(directory/(n+'.npy'),mmap_mode='r') for n in
                ('positions','velocities','days','earth','earth_velocity')]
        edges,counts=slow_pairs(c.tracks,*arrays,progress=lambda s:print(s,flush=True))
        save_array(directory/'parallax.npy',edges)
        counts['implementation_sha256']=sha256(Path(__file__))
        save_json(marker,counts)
    if not (directory/'extended-csr.json').exists():
        parts=sorted(directory.glob('part-[0-9][0-9][0-9].npy'))
        edges=unique_edges(np.concatenate([np.load(p) for p in parts]+[np.load(directory/'parallax.npy')]))
        offsets,neighbors,scores=csr_graph(edges,len(c.tracks))
        for n,a in [('offsets',offsets),('neighbors',neighbors),('scores',scores)]:save_array(directory/('extended-'+n+'.npy'),a)
        save_json(directory/'extended-csr.json',dict(edges=len(edges),parallax_sha256=sha256(directory/'parallax.npy'),
            complete=True,reached_tracklets=int(np.sum(np.diff(offsets)>0))))
    print(marker.read_text(),flush=True)
