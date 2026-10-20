"""Validated reception-coordinate flow for the segment transform.

The enclosure is a Picard inclusion, not a collection of propagated samples.
Bounds concern exact arithmetic with the binary64 model constants supplied.
Ordinary numerical orbit propagation elsewhere is not thereby certified.
"""

import math

import numpy as np
from flint import acb, arb, arb_mat

from magictools.orbits import C_AU_DAY, GM
from research.detection_limit.interval_orbits import lower_float, upper_float
from .segment_composition import ReceptionChart


def _real(x):
    return x.real if isinstance(x, acb) else x


def _sqrt_positive(x):
    # A right-half-plane radicand provides a single holomorphic square root.
    if not _real(x) > 0:
        raise ValueError("Cannot separate a square-root branch from zero")
    return x.sqrt()


def _square(x):
    """Retain the range of a real square, including balls crossing zero."""
    if isinstance(x, acb):
        return acb(_square(x.real)-_square(x.imag), 2*x.real*x.imag)
    lo, hi = x.lower(), x.upper()
    upper = max(abs(lo).upper(), abs(hi).upper())
    lower = min(abs(lo).lower(), abs(hi).lower()) if not x.contains(0) else arb(0)
    return (lower*lower).union(upper*upper)


class CircularObserver:
    """Analytic circular observer, in fixed inertial tangent axes."""

    axes = np.eye(3)

    def __init__(self, rate=None):
        self.rate = float(np.sqrt(GM["sun"]) if rate is None else rate)
        if not np.isfinite(self.rate) or abs(self.rate) >= C_AU_DAY:
            raise ValueError("Finite subluminal circular observer required")
        self.dual_axes = [[arb(int(i == j)) for j in range(3)] for i in range(3)]

    def jet(self, time):
        t = time if isinstance(time, (arb, acb)) else arb(float(time))
        w = arb(self.rate)
        c, s = (w*t).cos(), (w*t).sin()
        z = t*0
        return [c, s, z], [-w*s, w*c, z], [-w*w*c, -w*w*s, z]

    def chart(self, time):
        p, v, _ = self.jet(arb(float(time)))
        return ReceptionChart([float(x) for x in p], [float(x) for x in v], *self.axes)


class PolynomialObserver:
    """Fixed-axis polynomial ephemeris, with ascending inertial coefficients.

    This defines a model; it does not certify that the polynomial matches a
    real observatory between recorded epochs.
    """

    def __init__(self, coefficients, axes=np.eye(3), epoch=0.):
        self.coefficients = np.asarray(coefficients, float)
        self.axes = np.asarray(axes, float)
        self.epoch = float(epoch)
        if (self.coefficients.ndim != 2 or self.coefficients.shape[1] != 3
                or not len(self.coefficients) or not np.all(np.isfinite(self.coefficients))
                or not np.isfinite(self.epoch)):
            raise ValueError("Finite ascending (degree+1,3) coefficients required")
        ReceptionChart([0., 0., 0.], [0., 0., 0.], *self.axes)
        inverse = arb_mat(self.axes.T.tolist()).inv()
        self.dual_axes = [[inverse[i, j] for j in range(3)] for i in range(3)]

    def jet(self, time):
        t = time-self.epoch
        result = []
        for derivative in range(3):
            vector = []
            for axis in range(3):
                total = t*0
                for j in range(len(self.coefficients)-1, derivative-1, -1):
                    factor = math.factorial(j)//math.factorial(j-derivative)
                    total = total*t+arb(float(self.coefficients[j, axis]))*factor
                vector.append(total)
            result.append(vector)
        return result

    def chart(self, time):
        p, v, _ = self.jet(arb(float(time)))
        return ReceptionChart([float(x) for x in p], [float(x) for x in v], *self.axes)


def rhs(time, q, observer):
    """Exact retarded two-body ODE in (rho,eta,wy,x,y,wx).

    Works on real Arb boxes or complex Acb polydisks/rectangles. All axes are
    fixed in inertial space. The observer supplies position, velocity,
    acceleration at reception time; source acceleration is at emission time.
    """
    rho, eta, wy, x, y, wx = q
    if not _real(rho) > 0:
        raise ValueError("Inverse range crosses zero")
    n, e, b = observer.axes
    O, V, A = observer.jet(time)
    D = [float(nj)+x*float(ej)+y*float(bj) for nj, ej, bj in zip(n, e, b)]
    length = _sqrt_positive(sum(_square(d) for d in D))
    line = [d/length for d in D]
    w = [(eta*d+wx*float(ej)+wy*float(bj))/rho
         for d, ej, bj in zip(D, e, b)]
    radial = sum(a*c for a, c in zip(line, w))
    alpha = 1-radial/C_AU_DAY
    if not _real(alpha) > 0:
        raise ValueError("Emission time is not increasing")
    velocity = [(v+u)/alpha for v, u in zip(V, w)]
    # Stronger than necessary on complex boxes, but also proves the real
    # source branch stays subluminal on every accepted tube.
    speed_upper = sum(abs(v).upper()**2 for v in velocity).sqrt()
    if not speed_upper < arb(C_AU_DAY):
        raise ValueError("Cannot certify subluminal source")
    position = [o+d/rho for o, d in zip(O, D)]
    radius2 = sum(_square(r) for r in position)
    radius = _sqrt_positive(radius2)
    # Computing radius2*sqrt(radius2) as two dependent broad balls can
    # unnecessarily cross zero. A single analytic power retains positivity.
    inverse_cube = radius2**arb("-1.5")
    acceleration = [-arb(GM["sun"])*r*inverse_cube for r in position]
    distance = length/rho
    transverse = sum(_square(u) for u in w)-_square(radial)
    denominator = C_AU_DAY+sum(a*v for a, v in zip(line, velocity))
    if not _real(denominator) > 0:
        raise ValueError("Retarded map derivative crosses zero")
    alpha_prime = -(transverse/distance
                    +sum(a*(g*alpha*alpha-oa)
                         for a, g, oa in zip(line, acceleration, A)))/denominator
    w_prime = [g*alpha*alpha+v*alpha_prime-oa
               for g, v, oa in zip(acceleration, velocity, A)]
    # Use the enclosed dual basis rather than assuming that decimal/double
    # tangent axes are exactly orthonormal in real arithmetic.
    dual_n, dual_e, dual_b = observer.dual_axes
    axial = sum(a*j for a, j in zip(w_prime, dual_n))*rho
    east = sum(a*j for a, j in zip(w_prime, dual_e))*rho
    north = sum(a*j for a, j in zip(w_prime, dual_b))*rho
    result = [-rho*eta, axial-eta*eta, north-y*axial-2*eta*wy,
              wx, wy, east-x*axial-2*eta*wx]
    if not all(z.is_finite() for z in result):
        raise ValueError("Nonfinite reception flow")
    return result


def _inflate_real(x):
    return arb(x.mid(), x.rad()*arb("1.15")+arb("1e-28"))


def _inflate(x):
    if isinstance(x, acb):
        return acb(_inflate_real(x.real),
                   arb(0) if x.imag.is_zero() else _inflate_real(x.imag))
    return _inflate_real(x)


def _inside(outer, inner):
    if isinstance(outer, acb):
        return (_inside(outer.real, inner.real)
                and (outer.imag.is_zero() and inner.imag.is_zero()
                     or _inside(outer.imag, inner.imag)))
    return outer.contains_interior(inner)


def _union(a, b):
    if isinstance(a, acb):
        return acb(a.real.union(b.real), a.imag.union(b.imag))
    return a.union(b)


def _expand(a, b):
    if isinstance(a, acb):
        return acb(_expand(a.real, b.real), _expand(a.imag, b.imag))
    if a.is_zero() and b.is_zero() or a.contains_interior(b):
        return a
    return _inflate_real(a.union(b))


def step(q, time, offset, observer):
    """Enclose one flow step and its entire path by Picard inclusion."""
    time, offset = arb(time), arb(offset)
    if not time.is_exact() or not offset.is_exact():
        raise ValueError("Exact time and offset required")
    span = arb(0).union(offset)
    time_span = time+span
    derivative = rhs(time_span, q, observer)
    tube = [_inflate(y+span*d) for y, d in zip(q, derivative)]
    for iteration in range(100):
        derivative = rhs(time_span, tube, observer)
        image = [y+span*d for y, d in zip(q, derivative)]
        if all(_inside(a, b) for a, b in zip(tube, image)):
            endpoint = [y+offset*d for y, d in zip(q, derivative)]
            return endpoint, tube, iteration+1
        tube = [_expand(a, b) for a, b in zip(tube, image)]
    raise ValueError("Picard inclusion failed; reduce the time step/domain")


def enclose(q, start, stop, observer, *, max_step=.25):
    """Cover a flow segment, including complex initial data if supplied."""
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("Positive finite maximum step required")
    start, stop = arb(start), arb(stop)
    if not start.is_exact() or not stop.is_exact() or not start.is_finite() or not stop.is_finite():
        raise ValueError("Finite exact endpoints required")
    count = max(1, math.ceil(upper_float(abs(stop-start))/max_step))
    # A power-of-two count makes every step an exact dyadic for float times.
    count = 1 << (count-1).bit_length()
    h = (stop-start)/count
    result, hull, iterations = list(q), list(q), 0
    for j in range(count):
        result, tube, used = step(result, start+j*h, h, observer)
        hull = [_union(a, b) for a, b in zip(hull, tube)]
        iterations += used
    return result, hull, {"steps": count, "picard_iterations": iterations}


def box_balls(center, widths):
    center, widths = np.asarray(center, float), np.asarray(widths, float)
    if (center.shape != (6,) or widths.shape != (6,)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(widths))
            or np.any(widths <= 0)):
        raise ValueError("Finite six-dimensional box with positive widths required")
    return [arb(float(c), float(w)) for c, w in zip(center, widths)]


def float_box(balls):
    """Return a float center/radius box proved to contain the supplied balls."""
    center = np.array([float(x.mid()) for x in balls])
    widths = np.array([upper_float(abs(x-arb(float(c)))) for x, c in zip(balls, center)])
    return center, widths


def reception_cover(old_chart, center, widths):
    """Enclose an entire old emitted-rate chart in reception coordinates."""
    if not old_chart.retarded_epoch:
        raise ValueError("An emission-anchored ObserverChart is required")
    rho, eta, wy, x, y, wx = box_balls(center, widths)
    n, e, b = old_chart.axes
    D = [float(nj)+x*float(ej)+y*float(bj) for nj, ej, bj in zip(n, e, b)]
    length = _sqrt_positive(sum(_square(d) for d in D))
    line = [d/length for d in D]
    av = [eta*d+wx*float(ej)+wy*float(bj) for d, ej, bj in zip(D, e, b)]
    dot_av = sum(a*v for a, v in zip(D, av))
    denominator = (rho*length*C_AU_DAY
                   +rho*sum(a*float(v) for a, v in zip(D, old_chart.velocity))+dot_av)
    if not denominator > 0:
        raise ValueError("Retarded derivative crosses zero")
    beta = -dot_av/denominator  # alpha-1, without subtracting nearly equal balls
    inverse = arb_mat(old_chart.axes.T.tolist()).inv()
    vn, ve, vb = [sum(inverse[i, j]*float(old_chart.velocity[j]) for j in range(3))
                  for i in range(3)]
    result = [rho, eta+beta*(rho*vn+eta),
              wy+beta*(rho*(vb-y*vn)+wy), x, y,
              wx+beta*(rho*(ve-x*vn)+wx)]
    return float_box(result)


def analytic_bound(center, widths, epoch, times, observer, frequency_bounds,
                   ellipse=1.5, max_step=.25):
    """Certify a tensor Chebyshev error ingredient, without sampled maxima.

    For each coordinate separately, enlarge that real interval to its complex
    Bernstein ellipse, cover all real observation times, and bound
    |exp(i k.theta)| by exp(sum |k_i| |Im theta_i|).
    All other coordinates remain real. Returned M_i bounds are per unit l1
    norm of the source coefficients. This can be conservative or fail.
    """
    if not np.isfinite(ellipse) or ellipse <= 1:
        raise ValueError("Bernstein ellipse parameter must exceed one")
    times = np.asarray(times, float)
    frequencies = np.asarray(frequency_bounds, float)
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or frequencies.shape != (2,) or np.any(frequencies < 0)
            or not np.all(np.isfinite(frequencies))):
        raise ValueError("Finite times and two nonnegative frequency bounds required")
    r = arb(float(ellipse))
    a, b = (r+1/r)/2, (r-1/r)/2
    bounds, work = [], 0
    for axis in range(6):
        initial = [acb(x) for x in box_balls(center, widths)]
        initial[axis] = acb(arb(float(center[axis]), (a*float(widths[axis])).upper()),
                            arb(0, (b*float(widths[axis])).upper()))
        hull = initial
        for endpoint in [float(np.min(times)), float(np.max(times))]:
            _, tube, cost = enclose(initial, float(epoch), endpoint, observer, max_step=max_step)
            hull = [_union(x, y) for x, y in zip(hull, tube)]
            work += cost["steps"]
        exponent = sum(arb(float(k))*abs(hull[j].imag) for k, j in zip(frequencies, [3, 4]))
        bounds.append(upper_float(exponent.exp()))
    return {"ellipse": float(ellipse), "kernel_modulus_bounds": bounds,
            "validated_flow_steps": work, "whole_box_exact_arithmetic": True,
            "floating_point_evaluation_certified": False}


def interpolation_bound(orders, certificate):
    """Conservative telescoping tensor Chebyshev interpolation bound."""
    orders = tuple(int(p) for p in orders)
    if len(orders) != 6 or any(p < 1 for p in orders):
        raise ValueError("Six positive interpolation orders required")
    r = arb(certificate["ellipse"])
    bound, amplification = arb(0), arb(1)
    for p, modulus in zip(orders, certificate["kernel_modulus_bounds"]):
        bound += amplification*4*arb(modulus)*r**(-p)/(1-1/r)
        amplification *= arb(2*p-1).sqrt()
    return upper_float(bound), upper_float(amplification)
