"""Interval fifth derivatives of retarded Kepler geometry.

Univariate jets carry derivatives / factorial in one initial-state direction;
their constant coefficient encloses the WHOLE initial-state box. A validated
Taylor integrator encloses the extended variational ODE. Implicit light-time
derivatives are obtained by triangular coefficient solves, not by treating
emission time as independent of the orbit. Established interval/AD methods.
This isolated fifth-order prototype retains the third-order engine unchanged.
"""

import math

from flint import arb

from research.detection_limit import interval_orbits as flow


class Jet:
    def __init__(self, values):
        self.c = [arb(value) for value in values] if isinstance(values, (list, tuple)) else [arb(values)]+[arb(0)]*5
        if len(self.c) != 6:
            raise ValueError("A fifth-order jet has six coefficients")

    @staticmethod
    def cast(value):
        return value if isinstance(value, Jet) else Jet(value)

    def __add__(self, other):
        other = self.cast(other)
        return Jet([a+b for a, b in zip(self.c, other.c)])

    __radd__ = __add__

    def __neg__(self):
        return Jet([-x for x in self.c])

    def __sub__(self, other):
        return self+-self.cast(other)

    def __rsub__(self, other):
        return self.cast(other)+-self

    def __mul__(self, other):
        other = self.cast(other)
        return Jet([sum((self.c[k]*other.c[n-k] for k in range(n+1)), arb(0)) for n in range(6)])

    __rmul__ = __mul__

    def reciprocal(self):
        if self.c[0].contains(0):
            raise ValueError("Jet denominator includes zero")
        result = [1/self.c[0]]
        for n in range(1, 6):
            result.append(-sum((self.c[k]*result[n-k] for k in range(1, n+1)), arb(0))/self.c[0])
        return Jet(result)

    def __truediv__(self, other):
        return self*self.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return self.cast(other)*self.reciprocal()

    def sqrt(self):
        if not self.c[0] > 0:
            raise ValueError("Jet square root not separated from zero")
        result = [self.c[0].sqrt()]
        for n in range(1, 6):
            result.append((self.c[n]-sum((result[k]*result[n-k] for k in range(1, n)), arb(0)))/(2*result[0]))
        return Jet(result)


def acceleration(position, mu=flow.MU, origin=None):
    if origin is not None:
        position = [s+o for s, o in zip(position, origin)]
    square = sum((x*x for x in position), Jet(0))
    factor = -Jet.cast(mu)/(square*square.sqrt())
    return [factor*x for x in position]


def inflate(jet):
    return Jet([arb(x.mid(), x.rad()*arb("1.25")+arb("1e-28")) for x in jet.c])


def tube(state, h, mu=flow.MU, origin=None):
    span = arb(0).union(h)
    derivative = state[3:]+acceleration(state[:3], mu, origin)
    enclosure = [inflate(y+d*span) for y, d in zip(state, derivative)]
    for _ in range(40):
        derivative = enclosure[3:]+acceleration(enclosure[:3], mu, origin)
        image = [y+d*span for y, d in zip(state, derivative)]
        if all(all(a.contains_interior(b) for a, b in zip(t.c, z.c)) for t, z in zip(enclosure, image)):
            return enclosure
        enclosure = [Jet([a if a.contains_interior(b) else
                          arb(a.union(b).mid(), a.union(b).rad()*arb("1.25")+arb("1e-28"))
                          for a, b in zip(t.c, z.c)]) for t, z in zip(enclosure, image)]
    raise ValueError("Could not close extended variational Picard tube")


def series(state, order, mu=flow.MU, origin=None):
    r = [[state[axis], state[axis+3]]+[Jet(0) for _ in range(order-1)] for axis in range(3)]
    if origin is not None:
        for axis in range(3):
            r[axis][0] += origin[axis]
    q, g = [], []
    exponent = arb(-3)/2
    for n in range(order-1):
        q.append(sum((r[a][j]*r[a][n-j] for a in range(3) for j in range(n+1)), Jet(0)))
        if n == 0:
            g.append(1/(q[0]*q[0].sqrt()))
        else:
            g.append(sum((q[j]*g[n-j]*((exponent+1)*j-n) for j in range(1, n+1)), Jet(0))/(q[0]*n))
        for a in range(3):
            r[a][n+2] = sum((r[a][j]*g[n-j] for j in range(n+1)), Jet(0))*(-Jet.cast(mu)/((n+1)*(n+2)))
    if origin is not None:
        for axis in range(3):
            r[axis][0] = state[axis]
    return r


def step(state, h, order=8, mu=flow.MU, origin=None):
    """Time-Taylor enclosure of every coefficient of the variational solution."""
    h = arb(h)
    enclosure = tube(state, h, mu, origin)
    coefficients, remainder = series(state, order+1, mu, origin), series(enclosure, order+2, mu, origin)
    powers = [arb(1)]
    for _ in range(order+1):
        powers.append(powers[-1]*h)
    position, velocity = [], []
    for a in range(3):
        position.append(sum((coefficients[a][j]*powers[j] for j in range(order+1)), Jet(0))
                        +remainder[a][order+1]*powers[order+1])
        velocity.append(sum((coefficients[a][j+1]*((j+1)*powers[j]) for j in range(order+1)), Jet(0))
                        +remainder[a][order+2]*((order+2)*powers[order+1]))
    result = position+velocity
    if not all(x.is_finite() for jet in result for x in jet.c):
        raise ValueError("Nonfinite variational flow")
    return result


def propagate(state, time, max_step=1., mu=flow.MU, origin=None):
    time = arb(time)
    count = max(1, int(math.ceil(flow.upper_float(abs(time))/max_step)))
    for _ in range(count):
        state = step(state, time/count, mu=mu, origin=origin)
    return state


def retarded_jets(initial, time, observer, light_time_max=1., *, mu=flow.MU,
                  light_speed=flow.C_AU_DAY, path_offset=None, origin=None):
    mu, light_speed = Jet.cast(mu), Jet.cast(light_speed)
    if not mu.c[0] > 0 or not light_speed.c[0] > 0:
        raise ValueError("Positive gravitational parameter and light speed required")
    received = propagate(initial, time, mu=mu, origin=origin)
    received_scalar = [jet.c[0] for jet in received]
    observer = [Jet.cast(float(x) if isinstance(x, (float, int)) else x) for x in observer]
    delay = arb(-light_time_max).union(arb(light_time_max if path_offset is not None else 0))
    offset = Jet(0) if path_offset is None else Jet.cast(path_offset)
    scalar_origin = [o.c[0] for o in origin] if origin is not None else None
    absolute_scalar = ([s+o for s, o in zip(received_scalar[:3], scalar_origin)]+received_scalar[3:]
                       if scalar_origin is not None else received_scalar)
    enclosing = flow._tube(absolute_scalar, delay, mu.c[0])
    if not flow._norm(enclosing[3:]) < light_speed.c[0]:
        raise ValueError("Cannot certify unique retarded root")
    for iteration in range(12):
        if origin is None:
            scalar = flow.step(received_scalar, delay, mu=mu.c[0])
        else:
            from .translated_flow import scalar_step
            scalar = scalar_step(received_scalar, delay, scalar_origin, mu.c[0])
        updated = (offset.c[0]-flow._norm([x-o.c[0] for x, o in zip(scalar[:3], observer)]))/light_speed.c[0]
        if iteration == 0 and not delay.contains(updated):
            raise ValueError("Retarded interval is not invariant")
        delay = delay.intersection(updated)
    emitted = step(received, delay, mu=mu, origin=origin)
    q = [r-o for r, o in zip(emitted[:3], observer)]
    v = emitted[3:]
    # Through fifth order in the implicit emission-time displacement.
    # The recurrence also includes all parameter derivatives of each time
    # coefficient; a cubic jerk expansion would be invalid here.
    time_coefficients = series(emitted, 5, mu, origin)
    distance0 = flow._norm([jet.c[0] for jet in q])
    if not distance0 > 0:
        raise ValueError("Retarded box may intersect observer")
    denominator = light_speed.c[0]+sum((r.c[0]*vel.c[0] for r, vel in zip(q, v)), arb(0))/distance0
    if not denominator > 0:
        raise ValueError("Light-time implicit derivative is singular")
    dh = Jet(0)

    def composition():
        powers = [Jet(1)]
        for _ in range(5):
            powers.append(powers[-1]*dh)
        return [sum((coefficient*power for coefficient, power in zip(row, powers)), Jet(0))-o
                for row, o in zip(time_coefficients, observer)]

    for order in range(1, 6):
        composed = composition()
        distance = sum((r*r for r in composed), Jet(0)).sqrt()
        # In scaled coordinates c*rho also depends on the chart parameter.
        # Its derivatives times the (enclosed) delay must enter the implicit
        # equation. For constant c this reduces to the original recurrence.
        residual = light_speed*(Jet(delay)+dh)+distance-offset
        dh.c[order] = -residual.c[order]/denominator
    return composition()


def projected_fifth_coefficients(box, direction, time, observer, normal, east, north):
    initial = [Jet([value, float(derivative), 0, 0, 0, 0]) for value, derivative in zip(box, direction)]
    q = retarded_jets(initial, time, observer)
    dot = lambda axis: sum((r*float(a) for r, a in zip(q, axis)), Jet(0))
    denominator = dot(normal)
    if not denominator.c[0] > 0:
        raise ValueError("Variational cell crosses tangent horizon")
    return [(dot(axis)/denominator).c[5] for axis in (east, north)]
