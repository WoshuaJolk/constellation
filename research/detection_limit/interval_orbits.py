"""Validated short-arc two-body flow and pixel means using Arb balls.

Taylor coefficients at the initial interval give the polynomial; coefficients
over a Picard-enclosing tube give the Lagrange remainder. No sampled derivative
is promoted to a bound. This is deliberately conservative and may refuse a cell.
The specified model is heliocentric point-mass dynamics with fixed observers,
gnomonic WCS and independent, known Gaussian pixel noise.
"""

from dataclasses import dataclass
import math

import numpy as np
from flint import arb, ctx

from .certificates import GaussianCover


MU = "0.0002959122082855911025"  # Exact parameter of this model, AU^3/day^2.
C_AU_DAY = "173.144632674240"     # Exact parameter of this model.


def upper_float(value):
    """Round an Arb upper endpoint outward to binary64."""
    result = float(arb(value).upper())
    if not math.isfinite(result):
        raise ValueError("Nonfinite interval upper endpoint")
    return float(np.nextafter(result, np.inf))


def lower_float(value):
    result = float(arb(value).lower())
    if not math.isfinite(result):
        raise ValueError("Nonfinite interval lower endpoint")
    return float(np.nextafter(result, -np.inf))


def _norm(vector):
    # python-flint 0.8's generic ** can return nan for balls containing zero.
    square = sum((x*x for x in vector), arb(0))
    if square > 0:
        return square.sqrt()
    return arb(0).union(square.upper().sqrt())


def _acceleration(position, mu):
    square = sum((x*x for x in position), arb(0))
    if not square > 0:
        raise ValueError("Orbit tube may intersect the central singularity")
    scale = -mu / (square * square.sqrt())
    return [scale*x for x in position]


def _inflate(x, factor="1.2"):
    return arb(x.mid(), x.rad()*arb(factor)+arb("1e-25"))


def _tube(state, h, mu):
    span = arb(0).union(h)
    derivative = state[3:] + _acceleration(state[:3], mu)
    tube = [_inflate(y + span*d) for y,d in zip(state,derivative)]
    for _ in range(30):
        derivative = tube[3:] + _acceleration(tube[:3], mu)
        image = [y+span*d for y,d in zip(state,derivative)]
        if all(t.contains_interior(z) for t,z in zip(tube,image)):
            return tube
        tube = [t if t.contains_interior(z) else _inflate(t.union(z)) for t,z in zip(tube,image)]
    raise ValueError("Could not validate a collision-free Picard tube; split time/cell")


def _position_series(state, order, mu):
    """Taylor coefficients through order for r''=-mu*r/(r.r)^(3/2)."""
    r = [[state[axis],state[axis+3]]+[arb(0)]*(order-1) for axis in range(3)]
    q, g = [], []
    exponent = arb(-3)/2
    for n in range(order-1):
        q.append(sum((r[a][j]*r[a][n-j] for a in range(3) for j in range(n+1)), arb(0)))
        if n == 0:
            # For interval boxes crossing an axis, sum of products can be too
            # wide; compute the constant coefficient as a sum of squares.
            q[0] = sum((r[a][0]*r[a][0] for a in range(3)), arb(0))
            if not q[0] > 0:
                raise ValueError("Taylor coefficient interval contains a collision")
            g.append(1/(q[0]*q[0].sqrt()))
        else:
            g.append(sum((((exponent+1)*j-n)*q[j]*g[n-j] for j in range(1,n+1)),arb(0))/(n*q[0]))
        for a in range(3):
            r[a][n+2] = -mu*sum((r[a][j]*g[n-j] for j in range(n+1)),arb(0))/((n+1)*(n+2))
    return r


def step(state, h, *, order=10, mu=MU):
    """Enclose the solution at every time in interval h relative to state."""
    if not isinstance(order,int) or not 2 <= order <= 20:
        raise ValueError("Taylor order must be an integer in [2,20]")
    state, h, mu = [arb(x) for x in state], arb(h), arb(mu)
    if len(state) != 6 or not all(x.is_finite() for x in state) or not h.is_finite() or not mu > 0:
        raise ValueError("Finite six-dimensional state/time and positive mu required")
    tube = _tube(state,h,mu)
    coefficients = _position_series(state,order+1,mu)
    remainder = _position_series(tube,order+2,mu)
    powers = [arb(1)]
    for _ in range(order+1):
        powers.append(powers[-1]*h)
    position,velocity = [],[]
    for a in range(3):
        position.append(sum((coefficients[a][j]*powers[j] for j in range(order+1)),arb(0))
                        +remainder[a][order+1]*powers[order+1])
        velocity.append(sum(((j+1)*coefficients[a][j+1]*powers[j] for j in range(order+1)),arb(0))
                        +(order+2)*remainder[a][order+2]*powers[order+1])
    if not all(x.is_finite() for x in position+velocity):
        raise ValueError("Nonfinite validated flow")
    return position+velocity


def propagate(state, time, *, max_step=2.0, order=10):
    """Validated flow to a specified reception time; supports both time signs."""
    time = arb(time)
    if not time.is_exact() or not time.is_finite() or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("An exact finite time and positive max_step are required")
    count = max(1,int(math.ceil(upper_float(abs(time))/max_step)))
    result = [arb(x) for x in state]
    for _ in range(count):
        result = step(result,time/count,order=order)
    return result


def apparent(state, time, observer, *, light_time_max=1.0, order=10):
    """Enclose all retarded positions, with an invariant emission-time interval."""
    if not math.isfinite(light_time_max) or light_time_max <= 0:
        raise ValueError("Positive maximum light travel time required")
    at_reception = propagate(state,time,order=order)
    observer = [arb(x.item() if isinstance(x,np.generic) else x) for x in observer]
    if len(observer) != 3 or not all(x.is_finite() for x in observer):
        raise ValueError("Three finite observer coordinates required")
    emission = arb(-light_time_max).union(arb(0))
    tube = _tube(at_reception,emission,arb(MU))
    if not _norm(tube[3:]) < arb(C_AU_DAY):
        raise ValueError("Cannot establish uniqueness of light-time root")
    for iteration in range(12):
        positions = step(at_reception,emission,order=order)[:3]
        delay = -_norm([r-o for r,o in zip(positions,observer)])/arb(C_AU_DAY)
        if iteration == 0 and not emission.contains(delay):
            raise ValueError("Light-time interval not invariant; increase light_time_max")
        emission = emission.intersection(delay)
    positions = step(at_reception,emission,order=order)[:3]
    return positions, emission


@dataclass
class Exposure:
    time: float
    observer: tuple
    center: tuple
    east: tuple
    north: tuple
    pixel_scale_arcsec: float = 1.0
    sigma_pixels: float = 1.0
    shape: tuple = (7,7)
    flux_scale: float = 1.0
    noise_sigma: float = 1.0

    def __post_init__(self):
        for name in ("pixel_scale_arcsec","sigma_pixels","flux_scale","noise_sigma"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError("Exposure scales must be positive and finite")
        for vector in (self.observer,self.center,self.east,self.north):
            if len(vector) != 3 or not np.all(np.isfinite(vector)):
                raise ValueError("Exposure vectors must have three finite entries")
        for name in ("observer","center","east","north"):
            setattr(self,name,tuple(float(x) for x in getattr(self,name)))
        if not math.isfinite(self.time) or len(self.shape) != 2 or any(not isinstance(n,int) or n < 1 for n in self.shape):
            raise ValueError("Finite time and positive integer image shape required")


def gaussian_pixel(coordinate, left, right, sigma):
    """Enclose a pixel integral using its unimodality in source position."""
    width=arb(sigma)*arb(2).sqrt()
    value=lambda c: ((right-c)/width).erf()/2-((left-c)/width).erf()/2
    result=value(coordinate.lower()).union(value(coordinate.upper()))
    midpoint=(left+right)/2
    if coordinate.contains(midpoint):
        result=result.union(value(midpoint))
    return result


def pixel_means(state, exposures, flux, *, light_time_max=1.0):
    """Whitened, pixel-integrated Gaussian PSF means for an entire orbit cell.

    Flux is total source counts at unit exposure scale, not stamp-normalized.
    Cropped PSF wings remain missing; no renormalization hides lost information.
    Basis vectors define the supplied gnomonic WCS exactly as stored.
    """
    flux = arb(flux)
    if not flux >= 0:
        raise ValueError("Nonnegative flux required")
    means = []
    for exposure in exposures:
        position,_ = apparent(state,exposure.time,exposure.observer,light_time_max=light_time_max)
        direction = [x-arb(o) for x,o in zip(position,exposure.observer)]
        dot = lambda v: sum((x*arb(a) for x,a in zip(direction,v)),arb(0))
        denominator = dot(exposure.center)
        if not denominator > 0:
            raise ValueError("Orbit cell crosses the gnomonic horizon")
        scale = arb(exposure.pixel_scale_arcsec)*arb.pi()/648000
        x,y = dot(exposure.east)/denominator/scale, dot(exposure.north)/denominator/scale
        integrals = []
        for coordinate,n in ((x,exposure.shape[1]),(y,exposure.shape[0])):
            integrals.append([gaussian_pixel(coordinate,arb(2*j-n)/2,arb(2*j-n+2)/2,
                                             exposure.sigma_pixels) for j in range(n)])
        factor = flux*arb(exposure.flux_scale)/arb(exposure.noise_sigma)
        means.extend([factor*gx*gy for gy in integrals[1] for gx in integrals[0]])
    return means


def mean_ball(values):
    """Use binary64 representatives, charging their rounding to the radius."""
    midpoints = np.array([float(x.mid()) for x in values])
    errors = [abs(x-arb(float(m))).upper() for x,m in zip(values,midpoints)]
    radius = upper_float(sum((e**2 for e in errors),arb(0)).sqrt())
    return midpoints,radius


def uniform_boxes(center, half_widths, levels):
    """Exact dyadic partition of a six-dimensional box; no sampled gaps."""
    center,half_widths = [arb(x) for x in center],[arb(x) for x in half_widths]
    if len(center) != 6 or len(half_widths) != 6 or any(not x > 0 for x in half_widths):
        raise ValueError("Six positive half-widths required")
    if len(levels) != 6 or any(not isinstance(n,int) or not 0 <= n <= 12 for n in levels):
        raise ValueError("Six integer split levels in [0,12] required")
    count = 2**sum(levels)
    if count > 4096:
        raise ValueError("Research budget: at most 4096 cells")
    import itertools
    cells = []
    for index in itertools.product(*(range(2**level) for level in levels)):
        cells.append([arb(c-w+(2*j+1)*w/(2**level),w/(2**level))
                      for c,w,j,level in zip(center,half_widths,index,levels)])
    return cells


def orbit_cover(center, half_widths, levels, exposures, flux, *, light_time_max=1.0):
    """Uniform-prior cover; preserve interval precision independently of callers."""
    with ctx.workprec(96):
        cells = uniform_boxes(center,half_widths,levels)
        means,radii = [],[]
        for cell in cells:
            m,r = mean_ball(pixel_means(cell,exposures,flux,light_time_max=light_time_max))
            means.append(m)
            radii.append(r)
        return GaussianCover(np.array(means),np.array(radii),np.full(len(cells),1/len(cells)))
