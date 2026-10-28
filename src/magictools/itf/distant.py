"""Optional bound-orbit starting guesses for a distant-object research lane.

A proposal grid, not an orbital classification or an admissible-region proof:
finite-difference sky motion/observer motion and astrometric errors can exclude
valid distances from this approximate construction. Keep the ordinary Gauss,
all-conic range/radial starts and unresolved cases; never use this to reject them.
"""
import numpy as np
from .kernels import MU


def distant_seeds(t, units, observers, *, distances=None, radial_points=9):
    """Return extra fit_chain seeds plus per-distance proposal accounting.

    Sample heliocentrically bound velocities along the observed direction at
    each trial geocentric distance. The interval accounts for Earth's velocity;
    radial speed is not incorrectly assumed centered at zero. No energy cuts
    are added to final fitting, and the source data are never altered.
    """
    t, units, observers = (np.asarray(x,dtype=np.float64) for x in (t,units,observers))
    if t.ndim!=1 or len(t)<2 or units.shape!=(len(t),3) or observers.shape!=units.shape:
        raise ValueError('Expected times and (observations,3) directions/observers')
    if not all(np.isfinite(x).all() for x in (t,units,observers)) or np.any(np.diff(t)<0) or t[-1]<=t[0]:
        raise ValueError('Finite chronological observations with positive span required')
    if not np.allclose(np.linalg.norm(units,axis=1),1,atol=1e-10,rtol=0):
        raise ValueError('Directions must be unit vectors')
    if not isinstance(radial_points,int) or radial_points<2:
        raise ValueError('At least two radial samples required')
    distances=np.geomspace(30.,2000.,64) if distances is None else np.asarray(distances,dtype=float)
    if distances.ndim!=1 or not np.isfinite(distances).all() or np.any(distances<=0):
        raise ValueError('Distances must be a finite positive vector')
    mid=len(t)//2;epoch=float(t[mid]);dt=t[-1]-t[0];u=units[mid]
    seeds=[];rows=[]
    for distance in distances:
        position=observers[mid]+distance*u;radius=float(np.linalg.norm(position))
        base=(observers[-1]-observers[0]+distance*(units[-1]-units[0]))/dt
        center=-float(base@u);half_squared=center*center-float(base@base)+2*MU/radius
        row=dict(geocentric_distance_au=float(distance),heliocentric_radius_au=radius,
                 radial_center_au_day=center,radial_half_squared=float(half_squared),starts=0)
        if np.isfinite(half_squared) and half_squared>0:
            half=np.sqrt(half_squared)
            for fraction in np.linspace(-.95,.95,radial_points):
                velocity=base+(center+fraction*half)*u
                seeds.append((np.r_[position,velocity],epoch))
            row['starts']=radial_points;row['status']='proposed'
        else:
            row['status']='no_bound_proposal_from_chord_motion_not_an_exclusion'
        rows.append(row)
    return seeds,dict(distances=rows,starts=len(seeds),epoch_mjd_tdb=epoch,
        policy='Additional bound distant proposals only; preserve ordinary all-conic starts',
        limitations='Chord-motion approximation; no measurement uncertainty integration, completeness or rejection authority')
