"""Validated position increments in coordinates translated by a fixed origin."""

from flint import arb

from research.detection_limit import interval_orbits as flow


def scalar_step(state, h, origin, mu, order=10):
    """Integrate s''=-mu*(s+origin)/|s+origin|^3 without subtracting wide balls.

    Acceleration uses heliocentric coordinates. The constant position term is
    kept in relative coordinates, preventing artificial loss of the initial
    observer-position correlation.
    """
    h = arb(h)
    heliocentric = [s+o for s, o in zip(state[:3], origin)]+state[3:]
    tube = flow._tube(heliocentric, h, mu)
    coefficients = flow._position_series(heliocentric, order+1, mu)
    remainder = flow._position_series(tube, order+2, mu)
    powers = [arb(1)]
    for _ in range(order+1):
        powers.append(powers[-1]*h)
    position, velocity = [], []
    for axis in range(3):
        position.append(state[axis]+sum((coefficients[axis][j]*powers[j] for j in range(1, order+1)), arb(0))
                        +remainder[axis][order+1]*powers[order+1])
        velocity.append(sum((coefficients[axis][j+1]*(j+1)*powers[j] for j in range(order+1)), arb(0))
                        +remainder[axis][order+2]*(order+2)*powers[order+1])
    if not all(x.is_finite() for x in position+velocity):
        raise ValueError("Nonfinite translated interval flow")
    return position+velocity
