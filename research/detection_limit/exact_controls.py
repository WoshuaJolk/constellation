"""Independent analytic continuous experiments, evaluated with Arb arithmetic."""

from flint import arb, ctx

from .interval_orbits import lower_float, upper_float


def circle_power(radius, alpha="0.01", *, tolerance="1e-25"):
    """All directions in R^2, known amplitude, independent N(0,1) coordinates.

    Uniform angular LR is exp(-r^2/2) I0(r*||y||), increasing in ||y||.
    Radial rejection has equal power at every angle: both average and minimax
    optimal. Noncentral chi-square survival is enclosed by its positive Poisson
    series; the remaining Poisson probability bounds the omitted tail.
    This classical invariant-testing example is not a Kepler solution.
    """
    with ctx.workprec(128):
        r,a,tol = arb(radius),arb(alpha),arb(tolerance)
        if not r >= 0 or not (a > 0 and a < 1) or not (tol > 0 and tol < 1):
            raise ValueError("Nonnegative radius, alpha/tolerance in (0,1) required")
        noncentral,x = r*r/2,-a.log()
        poisson = (-noncentral).exp()
        poisson_sum = poisson
        gamma_term = (-x).exp()
        gamma_sum = gamma_term
        power = poisson*gamma_sum
        for k in range(1,10000):
            poisson *= noncentral/k
            poisson_sum += poisson
            gamma_term *= x/k
            gamma_sum += gamma_term
            power += poisson*gamma_sum
            remainder = 1-poisson_sum
            if remainder.upper() < tol:
                return {"power_interval":[max(0,lower_float(power)),min(1,upper_float(power+remainder))],
                        "power_arb":power.str(35), "tail_bound":upper_float(remainder),
                        "threshold_squared_radius":upper_float(2*x), "series_terms":k+1,
                        "objective":"both uniform-prior average and worst-case over every angle",
                        "arithmetic":"Arb 128-bit balls with positive-series tail bound"}
        raise ValueError("Poisson series budget exhausted")
