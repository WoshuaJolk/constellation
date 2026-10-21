"""Synthetic astrometry from analytic circular motion, independent of Kepler's propagator."""
import numpy as np
from scipy.optimize import brentq
from .orbits import GM, C_AU_DAY


def example():
    radius = 2.5
    frequency = np.sqrt(GM["sun"]/radius**3)
    earth_frequency = np.sqrt(GM["sun"])
    phases = [0., 1e-7, .015]
    def state(phase):
        return [radius*np.cos(phase), radius*np.sin(phase), 0.,
                -radius*frequency*np.sin(phase), radius*frequency*np.cos(phase), 0.]
    tracklets = []
    for label, reception, phase in [("A", 0., 0.), ("B", 18., 0.),
                                     ("C", 45., 0.), ("D", 18.04, .015)]:
        observations = []
        for j, t in enumerate([reception, reception+.02]):
            observer = np.array([np.cos(earth_frequency*t), np.sin(earth_frequency*t), 0.])
            def position(emitted):
                angle = phase+frequency*emitted
                return radius*np.array([np.cos(angle), np.sin(angle), 0.])
            emitted = brentq(lambda e: e+np.linalg.norm(position(e)-observer)/C_AU_DAY-t,
                             t-1., t, xtol=1e-13)
            direction = position(emitted)-observer
            observations.append(dict(
                id=label+str(j), observatory="synthetic-geocenter", mjd_tdb=60000.+t,
                ra_deg=float(np.rad2deg(np.arctan2(direction[1], direction[0])) % 360),
                dec_deg=0., sigma_arcsec=.3, observer_heliocentric_au=observer.tolist()))
        tracklets.append(dict(id=label, observations=observations))
    return dict(schema_version=1, snapshot_id="synthetic-circular-45-days",
                epoch_mjd_tdb=60000., frame="heliocentric_icrs", tracklets=tracklets,
                orbit_prior=[dict(id="orbit-"+str(i), state=state(p), weight=1.)
                             for i, p in enumerate(phases)],
                link_activity=1., queries=[["A", "B"], ["A", "C"], ["A", "D"]],
                max_seconds=180., max_nodes=100000, top_k=10,
                target_unresolved_mass=0.)
