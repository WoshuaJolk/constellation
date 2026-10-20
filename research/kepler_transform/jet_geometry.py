"""Interval third derivatives of retarded Kepler geometry.

Univariate jets carry derivatives / factorial in one initial-state direction;
their constant coefficient encloses the WHOLE initial-state box. A validated
Taylor integrator encloses the extended variational ODE. Implicit light-time
derivatives are obtained by triangular coefficient solves, not by treating
emission time as independent of the orbit. Established interval/AD methods.
"""

import math

from flint import arb

from research.detection_limit import interval_orbits as flow


class Jet:
    def __init__(self, values):
        self.c = [arb(value) for value in values] if isinstance(values, (list, tuple)) else [arb(values), arb(0), arb(0), arb(0)]
        if len(self.c) != 4:
            raise ValueError("A third-order jet has four coefficients")

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
        return Jet([sum((self.c[k]*other.c[n-k] for k in range(n+1)), arb(0)) for n in range(4)])

    __rmul__ = __mul__

    def reciprocal(self):
        if self.c[0].contains(0):
            raise ValueError("Jet denominator includes zero")
        result = [1/self.c[0]]
        for n in range(1, 4):
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
        for n in range(1, 4):
            result.append((self.c[n]-sum((result[k]*result[n-k] for k in range(1, n)), arb(0)))/(2*result[0]))
        return Jet(result)


def acceleration(position):
    square = sum((x*x for x in position), Jet(0))
    factor = -Jet(flow.MU)/(square*square.sqrt())
    return [factor*x for x in position]


def inflate(jet):
    return Jet([arb(x.mid(), x.rad()*arb("1.25")+arb("1e-28")) for x in jet.c])


def tube(state, h):
    span = arb(0).union(h)
    derivative = state[3:]+acceleration(state[:3])
    enclosure = [inflate(y+d*span) for y, d in zip(state, derivative)]
    for _ in range(40):
        derivative = enclosure[3:]+acceleration(enclosure[:3])
        image = [y+d*span for y, d in zip(state, derivative)]
        if all(all(a.contains_interior(b) for a, b in zip(t.c, z.c)) for t, z in zip(enclosure, image)):
            return enclosure
        enclosure = [Jet([a if a.contains_interior(b) else
                          arb(a.union(b).mid(), a.union(b).rad()*arb("1.25")+arb("1e-28"))
                          for a, b in zip(t.c, z.c)]) for t, z in zip(enclosure, image)]
    raise ValueError("Could not close extended variational Picard tube")


def series(state, order):
    r = [[state[axis], state[axis+3]]+[Jet(0) for _ in range(order-1)] for axis in range(3)]
    q, g = [], []
    exponent = arb(-3)/2
    for n in range(order-1):
        q.append(sum((r[a][j]*r[a][n-j] for a in range(3) for j in range(n+1)), Jet(0)))
        if n == 0:
            g.append(1/(q[0]*q[0].sqrt()))
        else:
            g.append(sum((q[j]*g[n-j]*((exponent+1)*j-n) for j in range(1, n+1)), Jet(0))/(q[0]*n))
        for a in range(3):
            r[a][n+2] = sum((r[a][j]*g[n-j] for j in range(n+1)), Jet(0))*(-arb(flow.MU)/((n+1)*(n+2)))
    return r


def step(state, h, order=8):
    """Time-Taylor enclosure of every coefficient of the variational solution."""
    h = arb(h)
    enclosure = tube(state, h)
    coefficients, remainder = series(state, order+1), series(enclosure, order+2)
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


def propagate(state, time, max_step=1.):
    time = arb(time)
    count = max(1, int(math.ceil(flow.upper_float(abs(time))/max_step)))
    for _ in range(count):
        state = step(state, time/count)
    return state


def retarded_jets(initial, time, observer, light_time_max=1.):
    received = propagate(initial, time)
    received_scalar = [jet.c[0] for jet in received]
    observer = [arb(float(x)) for x in observer]
    delay = arb(-light_time_max).union(arb(0))
    enclosing = flow._tube(received_scalar, delay, arb(flow.MU))
    if not flow._norm(enclosing[3:]) < arb(flow.C_AU_DAY):
        raise ValueError("Cannot certify unique retarded root")
    for iteration in range(12):
        scalar = flow.step(received_scalar, delay)
        updated = -flow._norm([x-o for x, o in zip(scalar[:3], observer)])/arb(flow.C_AU_DAY)
        if iteration == 0 and not delay.contains(updated):
            raise ValueError("Retarded interval is not invariant")
        delay = delay.intersection(updated)
    emitted = step(received, delay)
    q = [r-o for r, o in zip(emitted[:3], observer)]
    v, a = emitted[3:], acceleration(emitted[:3])
    radius2 = sum((r*r for r in emitted[:3]), Jet(0))
    radius = radius2.sqrt()
    rv = sum((r*vel for r, vel in zip(emitted[:3], v)), Jet(0))
    jerk = [(vel/(radius2*radius)-r*rv*3/(radius2*radius2*radius))*(-arb(flow.MU))
            for r, vel in zip(emitted[:3], v)]
    distance0 = flow._norm([jet.c[0] for jet in q])
    if not distance0 > 0:
        raise ValueError("Retarded box may intersect observer")
    denominator = arb(flow.C_AU_DAY)+sum((r.c[0]*vel.c[0] for r, vel in zip(q, v)), arb(0))/distance0
    if not denominator > 0:
        raise ValueError("Light-time implicit derivative is singular")
    dh = Jet(0)

    def composition():
        return [r+vel*dh+acc*dh*dh/2+j*dh*dh*dh/6 for r, vel, acc, j in zip(q, v, a, jerk)]

    for order in range(1, 4):
        composed = composition()
        distance = sum((r*r for r in composed), Jet(0)).sqrt()
        dh.c[order] = -distance.c[order]/denominator
    return composition()


def projected_third_coefficients(box, direction, time, observer, normal, east, north):
    initial = [Jet([value, float(derivative), 0, 0]) for value, derivative in zip(box, direction)]
    q = retarded_jets(initial, time, observer)
    dot = lambda axis: sum((r*float(a) for r, a in zip(q, axis)), Jet(0))
    denominator = dot(normal)
    if not denominator.c[0] > 0:
        raise ValueError("Variational cell crosses tangent horizon")
    return [(dot(axis)/denominator).c[3] for axis in (east, north)]
