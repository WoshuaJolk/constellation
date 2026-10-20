"""Complex-source analyticity bounds from regularized Kepler algebra.

Uniformly enclose a retarded Sundman root using a complex contraction disk.
First-order complex jets bound phase variation across an entire output cell.
This certifies sufficient interpolation hypotheses, not the implementation's
floating-point contractions or all detector-model assumptions.
"""

from math import factorial

import numpy as np
from flint import acb, arb, ctx

from magictools.orbits import GM, C_AU_DAY
from research.detection_limit.interval_orbits import upper_float


def square_real(x):
    if x.contains(0):
        return arb(0).union(abs(x).upper()**2)
    return (abs(x).lower()**2).union(abs(x).upper()**2)


def right_sqrt(z):
    """Principal sqrt with an imaginary bound proportional to Im(z).

    The generic rectangular acb sqrt can lose this property for wide real
    intervals. Re(z)>0 gives an explicit analytic branch and sharper formula.
    """
    if not z.real > 0:
        raise ValueError("Complex square-root argument leaves the right half-plane")
    modulus = (square_real(z.real)+square_real(z.imag)).sqrt()
    real = ((modulus+z.real)/2).sqrt()
    return acb(real, z.imag/(2*real))


class Dual:
    size = 7

    def __init__(self, value, derivative=None):
        self.v = acb(value)
        self.d = [acb(0) for _ in range(self.size)] if derivative is None else list(derivative)

    @classmethod
    def cast(cls, value):
        return value if isinstance(value, cls) else cls(value)

    def __add__(self, other):
        other = self.cast(other)
        return Dual(self.v+other.v, [a+b for a, b in zip(self.d, other.d)])

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, other):
        return self+-self.cast(other)

    def __rsub__(self, other):
        return self.cast(other)+-self

    def __mul__(self, other):
        other = self.cast(other)
        return Dual(self.v*other.v, [a*other.v+self.v*b for a, b in zip(self.d, other.d)])

    __rmul__ = __mul__

    def reciprocal(self):
        if self.v.contains(0):
            raise ValueError("Complex dual denominator contains zero")
        value = 1/self.v
        return Dual(value, [-a*value*value for a in self.d])

    def __truediv__(self, other):
        return self*self.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return self.cast(other)*self.reciprocal()

    def sqrt(self):
        value = right_sqrt(self.v)
        return Dual(value, [a/(2*value) for a in self.d])


def stumpff_dual(z, terms=16):
    """Entire C,S and first derivatives with explicit complex tail bounds."""
    magnitude = abs(z.v).upper()
    if not magnitude < arb(".5"):
        raise ValueError("Certified short-anomaly Stumpff range exceeded")
    power, previous = acb(1), acb(1)
    c, s, cp, sp = acb(0), acb(0), acb(0), acb(0)
    for k in range(terms):
        sign = (-1)**k
        c += sign*power/factorial(2*k+2)
        s += sign*power/factorial(2*k+3)
        if k:
            cp += sign*k*previous/factorial(2*k+2)
            sp += sign*k*previous/factorial(2*k+3)
        previous, power = power, power*z.v
    values = []
    for value, derivative, shift in [(c, cp, 2), (s, sp, 3)]:
        ratio = magnitude/((2*terms+shift+1)*(2*terms+shift+2))
        tail = magnitude**terms/factorial(2*terms+shift)/(1-ratio)
        dtail = terms*magnitude**(terms-1)/factorial(2*terms+shift)/(1-ratio*arb(terms+1)/terms)
        value += acb(arb(0, tail.upper()), arb(0, tail.upper()))
        derivative += acb(arb(0, dtail.upper()), arb(0, dtail.upper()))
        values.append(Dual(value, [derivative*d for d in z.d]))
    return values


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Dual(0))


def regularized_equation(chart, parameters, time, anomaly, *, compensated=True):
    """F(s)=emission_time-reception_time+(distance-reference_path)/(c*rho)."""
    rho = parameters[0]
    relative = chart.relative_initial(parameters)
    position = [s+rho*float(o) for s, o in zip(relative[:3], chart.observer)]
    velocity = relative[3:]
    mu = GM["sun"]*rho*rho*rho
    # Tighten the known positive real cubic value; derivatives retain their
    # enclosing jet arithmetic.
    if rho.v.imag.is_zero() and rho.v.real > 0:
        r = rho.v.real
        mu.v = acb(arb(GM["sun"])*(r.lower()**3).union(r.upper()**3))
    radius, rv = dot(position, position).sqrt(), dot(position, velocity)
    lam = dot(velocity, velocity)-2*mu/radius
    s2, s3 = anomaly*anomaly, anomaly*anomaly*anomaly
    c, h = stumpff_dual(-lam*s2)
    emitted = [
        r+radius*v*anomaly+(-mu*r/radius+rv*v)*s2*c+lam*radius*v*s3*h
        for r, v in zip(position, velocity)]
    elapsed = radius*anomaly+rv*s2*c+(lam*radius+mu)*s3*h
    regularized_radius = radius+rv*anomaly+(lam*radius+mu)*s2*c+lam*rv*s3*h
    if not regularized_radius.v.real > 0:
        raise ValueError("Regularized radius is not separated from collision")
    angle = arb(GM["sun"]).sqrt()*time
    observer = [angle.cos(), angle.sin(), acb(0)]
    delta = [r-rho*o for r, o in zip(emitted, observer)]
    distance = dot(delta, delta).sqrt()
    offset = dot(relative[:3], relative[:3]).sqrt() if chart.retarded_epoch else Dual(0)
    residual = elapsed-time+(distance-offset)/(C_AU_DAY*rho)
    if compensated:
        # On a retarded root, elapsed = time + (offset-distance)/(c*rho).
        # Substitute this BEFORE interval evaluation. The leading observer
        # velocity cancels analytically; using T(s) and t as independent
        # intervals would destroy that cancellation.
        angle_dual = Dual(angle)
        observer_c, observer_s = stumpff_dual(angle_dual*angle_dual)
        angular_speed = arb(GM["sun"]).sqrt()
        observer_residual = [
            Dual(float(chart.observer[0])-1)+float(chart.velocity[0])*time+angle_dual*angle_dual*observer_c,
            Dual(float(chart.observer[1]))+(acb(float(chart.velocity[1]))-angular_speed)*time
            +angle_dual*angle_dual*angle_dual*observer_s,
            Dual(float(chart.observer[2]))+float(chart.velocity[2])*time]
        eta, wy, wx = parameters[1], parameters[2], parameters[5]
        angular_velocity = [eta*d+wx*float(e)+wy*float(n)
                            for d, e, n in zip(relative[:3], chart.axes[1], chart.axes[2])]
        delay = (offset-distance)/(C_AU_DAY*rho)
        delta = [d+rho*o+av*time+w*delay-mu*(r/radius*s2*c+w*s3*h)
                 for d, o, av, w, r in zip(relative[:3], observer_residual,
                                          angular_velocity, velocity, position)]
    denominator = sum((r*float(a) for r, a in zip(delta, chart.axes[0])), Dual(0))
    if not denominator.v.real > 0:
        raise ValueError("Complex projection denominator leaves the right half-plane")
    projected = [sum((r*float(a) for r, a in zip(delta, axis)), Dual(0))/denominator
                 for axis in chart.axes[1:]]
    return residual, projected


def parameter_jets(center, widths, derivative_scales=None):
    result = []
    if derivative_scales is None:
        derivative_scales = widths
    for i, (c, w) in enumerate(zip(center, widths)):
        derivative = [acb(0) for _ in range(7)]
        derivative[i] = acb(float(derivative_scales[i]))
        result.append(Dual(arb(float(c), float(w)), derivative))
    return result


def anomaly_jet(value):
    derivative = [acb(0) for _ in range(7)]
    derivative[6] = acb(1)
    return Dual(value, derivative)


def global_speed_certificate(chart, center, widths):
    """Nonzero angular momentum and subluminal periapsis for the whole cell."""
    with ctx.workprec(128):
        q = [arb(float(c), float(w)) for c, w in zip(center, widths)]
        rho = q[0]
        if not rho > 0:
            raise ValueError("Inverse distance interval must be positive")
        state = chart.scaled_initial(q)
        r, v = state[:3], state[3:]
        cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                              a[0]*b[1]-a[1]*b[0]]
        def norm(a):
            value = sum((square_real(x) for x in a), arb(0))
            return value.sqrt() if value > 0 else arb(0).union(value.upper().sqrt())
        radius = norm(r)
        h = cross(r, v)
        angular = norm(h)
        if not radius > 0 or not angular > 0:
            raise ValueError("Global collision-free angular momentum bound failed")
        mu = arb(GM["sun"])*(rho.lower()**3).union(rho.upper()**3)
        eccentricity = norm([x/mu-y/radius for x, y in zip(cross(v, h), r)])
        speed = mu*(1+eccentricity)/angular
        ratio = speed/(arb(C_AU_DAY)*rho)
        if not ratio < 1:
            raise ValueError("Global subluminal periapsis bound failed")
        return {"nonzero_angular_momentum_certified": True,
                "global_periapsis_speed_over_light_speed_upper": upper_float(ratio),
                "unique_physical_retarded_root": True}


def root_and_gradients(chart, center, widths, time_center, time_half_width, rho=2., *,
                       compensated=True, derivative_scales=None):
    """Sufficient complex root/branch certificate for a complete real cell.

    The complex time rectangle encloses the Bernstein ellipse. Returns
    derivatives with respect to normalized coordinates of this output cell.
    """
    with ctx.workprec(128):
        rho = arb(float(rho))
        if not rho >= 1 or time_half_width <= 0:
            raise ValueError("Ellipse parameter >=1 and positive time half-width required")
        extent, height = (rho+1/rho)/2, (rho-1/rho)/2
        time = acb(arb(float(time_center), (arb(float(time_half_width))*extent).upper()),
                   arb(0, (arb(float(time_half_width))*height).upper()))
        nominal = parameter_jets(center, np.zeros(6))
        proposal = float(time_center)/(1+float(center[0]))
        for _ in range(8):
            f, _ = regularized_equation(chart, nominal, acb(float(time_center)), anomaly_jet(proposal), compensated=compensated)
            correction = f.v/f.d[6]
            proposal -= float(correction.real.mid())
            if upper_float(abs(correction)) < 1e-14:
                break
        f, _ = regularized_equation(chart, nominal, acb(float(time_center)), anomaly_jet(proposal), compensated=compensated)
        inverse = (1/f.d[6].mid()).mid()
        parameters = parameter_jets(center, widths, derivative_scales)
        center_value, _ = regularized_equation(chart, parameters, time, anomaly_jet(proposal), compensated=compensated)
        displacement = abs(inverse*center_value.v)
        radius = max(1e-12, 1.5*upper_float(displacement))
        for _ in range(16):
            enclosure = acb(arb(proposal, radius), arb(0, radius))
            residual, projected = regularized_equation(chart, parameters, time, anomaly_jet(enclosure), compensated=compensated)
            contraction = abs(1-inverse*residual.d[6])
            image_radius = displacement+contraction*arb(radius)
            if contraction < 1 and image_radius < arb(radius):
                if rho == 1:
                    # For real parameters/time, conjugation preserves the
                    # equation and this disk. Uniqueness makes its root real.
                    residual, projected = regularized_equation(
                        chart, parameters, time, anomaly_jet(acb(arb(proposal, radius))),
                        compensated=compensated)
                gradients = [
                    [p.d[j]-p.d[6]*residual.d[j]/residual.d[6] for j in range(6)]
                    for p in projected]
                if rho == 1:
                    gradients = [[acb(value.real) for value in axis] for axis in gradients]
                record = {
                    "certified": True, "ellipse_parameter": float(rho),
                    "time_center_days": float(time_center), "time_half_width_days": float(time_half_width),
                    "anomaly_disk_center": proposal, "anomaly_disk_radius": radius,
                    "contraction_upper": upper_float(contraction),
                    "image_radius_upper": upper_float(image_radius),
                    "method": "complex contraction disk; entire regularized dynamics; right-half-plane branches",
                    "retarded_observer_cancellation_preserved": compensated,
                    "regularized_radius_real_part_positive": True,
                    "connected_to_known_reference_emission_root": bool(
                        chart.retarded_epoch
                        and np.array_equal(chart.observer, np.array([1., 0., 0.]))
                        and abs(arb(float(time_center))) <= arb(float(time_half_width))*extent
                        and abs(arb(proposal)) < arb(radius)),
                    "positive_real_retarded_root_derivative": bool(
                        inverse.imag.is_zero() and inverse.real > 0 and contraction < 1),
                }
                return gradients, record
            radius = max(radius*1.5, 1.1*upper_float(image_radius))
        raise ValueError("Complex retarded root contraction did not close")


def phase_log_modulus_bound(gradients, frequency_center, frequency_half_width, frequency_scale, rho=2.):
    """Bound |Im(phi(x,z)-phi(center,z))| across the entire output cell."""
    with ctx.workprec(128):
        rho = arb(float(rho))
        extent, height = (rho+1/rho)/2, (rho-1/rho)/2
        frequencies = [
            arb(float(frequency_scale))*acb(
                arb(float(c), (arb(float(frequency_half_width))*extent).upper()),
                arb(0, (arb(float(frequency_half_width))*height).upper()))
            for c in frequency_center]
        result = sum((abs((frequencies[0]*gradients[0][j]+frequencies[1]*gradients[1][j]).imag)
                      for j in range(6)), arb(0))
        return upper_float(result)
