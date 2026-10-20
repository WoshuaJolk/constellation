"""Exact-arithmetic interpolation bounds for the complete segment graph."""

from flint import arb

from research.detection_limit.interval_orbits import upper_float
from .reception_flow import analytic_bound


def local_bound(grid, certificate):
    """Tensor Lobatto interpolation via Cauchy derivative bounds.

    A Bernstein ellipse contains a disk of radius w*(a-1) about EVERY point
    of its real interval, where a=(r+1/r)/2 and w is its half-width. For p
    Lobatto nodes in a cell of half-width h, the nodal polynomial is bounded
    by 2**(2-p)*h**p. Cauchy and the real interpolation remainder then bound
    each coordinate's error by 4*M*(h/(2*w*(a-1)))**p.

    The tensor telescoping sum charges preceding Lebesgue factors. A safe
    Lobatto factor is sqrt(2p-2), from discrete cosine orthogonality and
    Cauchy-Schwarz; for linear interpolation it is exactly one.
    """
    r = arb(certificate["ellipse"])
    halo = (r+1/r)/2-1
    error, amplification = arb(0), arb(1)
    for axis, cells, p in zip(grid.axes, grid.cells, grid.orders):
        lebesgue = arb(1) if p <= 2 else arb(2*p-2).sqrt()
        modulus = arb(certificate["kernel_modulus_bounds"][axis])
        local = 4*modulus*(1/(2*cells*halo))**p
        # A global Chebyshev truncation of degree p-1 is also reproduced on
        # every local cell. This bound decays with order even when the small
        # endpoint Cauchy disks do not cover a whole interpolation cell.
        global_best = (1+lebesgue)*2*modulus*r**(-p)/(1-1/r)
        chosen = min(upper_float(local), upper_float(global_best))
        error += amplification*arb(chosen)
        amplification *= lebesgue
    return upper_float(error), upper_float(amplification)


def certify(transform, *, ellipse=1.5, max_step=.25):
    """Certify a whole-root error per unit source l1 norm, or refuse.

    Includes every segment's local interpolation and all propagation of child
    errors. The certificate is usually conservative, sometimes prohibitively
    so. A finite bound is not a successful accuracy guarantee unless it meets
    the requested tolerance. Ordinary float matrix construction is excluded.
    """
    if transform.transfer_backend != "tensor":
        raise ValueError("Cascade mixed-coordinate domains require a separate certificate")
    if transform.frequencies is None:
        raise ValueError("Custom leaf models require their own analytic modulus certificate")
    records, steps = [], 0

    def visit(node):
        nonlocal steps
        child_bounds = [(0. if transform.exact_leaves and len(child.indices) == 1 else visit(child))
                        for child in node.children]
        frequencies = [abs(k).max(axis=0) for j, k in enumerate(transform.frequencies)
                       if j in node.indices]
        frequency_bound = [max(k[a] for k in frequencies) for a in range(2)]
        proof = analytic_bound(node.grid.center, node.grid.widths, node.epoch,
                               transform.times[node.indices], transform.observer,
                               frequency_bound, ellipse=ellipse, max_step=max_step)
        steps += proof["validated_flow_steps"]
        delta, amplification = local_bound(node.grid, proof)
        total = arb(delta)+arb(amplification)*arb(max(child_bounds, default=0.))
        result = upper_float(total)
        records.append({"epoch": node.epoch, "exposures": len(node.indices),
                        "local_error_per_source_l1": delta,
                        "lebesgue_upper": amplification,
                        "subtree_error_per_source_l1": result, "analytic": proof})
        return result

    error = visit(transform.root)
    return {"uniform_error_per_source_l1": error, "segments": records,
            "validated_flow_steps": steps,
            "includes_every_local_error_and_child_amplification": True,
            "exact_arithmetic_only": True, "floating_point_evaluation_certified": False,
            "full_objective_complete": False}
