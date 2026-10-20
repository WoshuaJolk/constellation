"""Validated scalar all-conic retarded positions using universal variables.

Newton iterations only propose a root. Acceptance requires interval endpoint
signs, a positive interval derivative, and a global subluminal speed bound
from nonzero angular momentum and periapsis. Entire Stumpff series have explicit
geometric remainder bounds. This is a certificate, not a convergence heuristic.

The short-anomaly implementation deliberately refuses |alpha*chi^2| >= 1/2;
the caller can use the slower validated ODE outside this computational regime.
Standard universal-variable identities, not a new dynamics result.
"""

from math import factorial

from flint import arb


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), arb(0))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def norm(a):
    value = sum((x**2 for x in a), arb(0))
    if not value >= 0:
        raise ValueError("Cannot enclose vector norm")
    return value.sqrt()


def stumpff(z, terms=16):
    magnitude = abs(z)
    if not magnitude < arb(".5"):
        raise ValueError("Validated Stumpff short-anomaly range exceeded")
    power, c, s = arb(1), arb(0), arb(0)
    for k in range(terms):
        c += power/factorial(2*k+2)
        s += power/factorial(2*k+3)
        power *= -z
    c_tail = magnitude**terms/factorial(2*terms+2)/(1-magnitude/((2*terms+3)*(2*terms+4)))
    s_tail = magnitude**terms/factorial(2*terms+3)/(1-magnitude/((2*terms+4)*(2*terms+5)))
    return c+arb(0, c_tail.upper()), s+arb(0, s_tail.upper())


def retarded_position(initial, time, observer, mu, speed, path_offset=0):
    """Enclose the unique physical retarded position relative to observer.

    Equation: speed*(elapsed - reception_time) + distance - path_offset = 0.
    A nonzero offset accounts for a reference emission epoch in scaled units.
    All inputs are Arb enclosures; each enclosed initial condition is covered.
    Wide inputs usually fail the tight root bracket and use the ODE fallback.
    """
    r, v = initial[:3], initial[3:]
    mu, speed, time, offset = map(arb, [mu, speed, time, path_offset])
    radius = norm(r)
    if not radius > 0 or not mu > 0 or not speed > 0:
        raise ValueError("Positive radius, gravity and light speed required")
    h = cross(r, v)
    angular = norm(h)
    if not angular > 0:
        raise ValueError("Nonzero angular momentum required for universal certificate")
    eccentricity = norm([a/mu-b/radius for a, b in zip(cross(v, h), r)])
    maximum_speed = mu*(1+eccentricity)/angular
    if not maximum_speed < speed:
        raise ValueError("Global periapsis speed bound is not subluminal")
    root_mu = mu.sqrt()
    rv = dot(r, v)/root_mu
    alpha = 2/radius-dot(v, v)/mu

    def evaluate(chi):
        c, s = stumpff(alpha*chi**2)
        elapsed = (rv*chi**2*c+(1-alpha*radius)*chi**3*s+radius*chi)/root_mu
        f, g = 1-chi**2*c/radius, elapsed-chi**3*s/root_mu
        position = [f*a+g*b for a, b in zip(r, v)]
        radial = norm(position)
        if not radial > 0:
            raise ValueError("Universal root bracket intersects central singularity")
        fdot = root_mu/(radial*radius)*(alpha*chi**3*s-chi)
        gdot = 1-chi**2*c/radial
        velocity = [fdot*a+gdot*b for a, b in zip(r, v)]
        delta = [a-b for a, b in zip(position, observer)]
        distance = norm(delta)
        if not distance > 0:
            raise ValueError("Universal root bracket intersects observer")
        residual = speed*(elapsed-time)+distance-offset
        derivative = radial/root_mu*(speed+dot(delta, velocity)/distance)
        return residual, derivative, delta

    reference_distance = norm([a-b for a, b in zip(r, observer)])
    proposed = (root_mu*(time+(offset-reference_distance)/speed)/radius).mid()
    for _ in range(8):
        residual, derivative, _ = evaluate(proposed)
        if not derivative > 0:
            raise ValueError("Cannot establish positive retarded anomaly derivative")
        correction = residual/derivative
        proposed = (proposed-correction).mid()
        if abs(correction) < arb("1e-30"):
            break
    bracket = arb(proposed, (1+abs(proposed))*arb("1e-25"))
    left, _, _ = evaluate(bracket.lower())
    right, _, _ = evaluate(bracket.upper())
    if not left < 0 or not right > 0:
        raise ValueError("Retarded anomaly bracket did not certify existence")
    _, derivative, delta = evaluate(bracket)
    if not derivative > 0:
        raise ValueError("Retarded anomaly bracket did not certify uniqueness")
    return delta
