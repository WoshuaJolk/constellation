"""Independent float64 pixel renderer used for controls and simulated truth.

It does not call the interval renderer, jet engine or quadrature implementation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtr


def render(exposures, sources, nuisance=None):
    nuisance = nuisance or {}
    images = []
    for e in exposures:
        t = e["time"]
        camera = e.get("instrument", "default")
        get = lambda k, default=0.: nuisance.get(camera+"_"+k, default)
        shape = np.asarray(e["data"]).shape
        yy, xx = np.indices(shape, dtype=float)
        xx -= (shape[1]-1)/2
        yy -= (shape[0]-1)/2
        image = np.full(shape, get("background", e.get("background", 0.)))
        psf = e["psf_sigma"]*np.exp(get("log_psf"))
        for source in sources:
            p = source["parameters"]
            if source["kind"] == "kepler":
                from research.detection_limit.interval_orbits import MU, C_AU_DAY
                mu, speed = float(MU), float(C_AU_DAY)
                state = np.array([p[k] for k in ("rx", "ry", "rz", "vx", "vy", "vz")])
                rhs = lambda time, s: np.r_[s[3:], -mu*s[:3]/np.linalg.norm(s[:3])**3]
                def at(time):
                    if time == 0:
                        return state[:3]
                    return solve_ivp(rhs, [0, time], state, rtol=2e-12, atol=1e-14, method="DOP853").y[:3, -1]
                observer = np.asarray(e["observer"])
                emission = brentq(lambda h: h+np.linalg.norm(at(t+h)-observer)/speed,
                                   -e.get("light_time_max", 1.), 0., xtol=1e-14)
                direction = at(t+emission)-observer
                scale = e["pixel_scale_arcsec"]*np.pi/648000
                x = direction@e["east"]/(direction@e["center"])/scale
                y = direction@e["north"]/(direction@e["center"])/scale
            else:
                x = p["x"]+p.get("vx", 0)*t+p.get("parallax", 0)*e.get("parallax_x", 0)
                y = p["y"]+p.get("vy", 0)*t+p.get("parallax", 0)*e.get("parallax_y", 0)
            x, y = np.array(e.get("affine", [[1, 0], [0, 1]]))@[x, y]+e.get("offset", [0, 0])
            x, y = x+get("dx"), y+get("dy")
            sx = np.hypot(psf, p.get("sigma_x", 0)) if source["kind"] == "galaxy" else psf
            sy = np.hypot(psf, p.get("sigma_y", 0)) if source["kind"] == "galaxy" else psf
            amplitude = p["flux"]*e.get("flux_scale", 1)*np.exp(
                get("log_gain")+p.get("color", 0)*e.get("band_coordinate", 0)+p.get("variability", 0)*t)
            image += amplitude*(ndtr((xx+.5-x)/sx)-ndtr((xx-.5-x)/sx))*(
                ndtr((yy+.5-y)/sy)-ndtr((yy-.5-y)/sy))
        images.append(image)
    return images
