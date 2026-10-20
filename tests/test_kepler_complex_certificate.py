"""Independent high-precision challenges to the interval proof components."""

import numpy as np
import pytest

flint = pytest.importorskip("flint")
mp = pytest.importorskip("mpmath")
from flint import acb, arb, ctx

from magictools.orbits import GM, C_AU_DAY
from research.kepler_transform.chart_experiments import setup
from research.kepler_transform.complex_source_certificate import (
    Dual, right_sqrt, root_and_gradients, stumpff_dual,
)
from research.kepler_transform.certify_global import choose_orders, partitioned_gradients


def contains_mp(ball, value):
    # Parse the independent value more accurately than the interval under
    # test; otherwise the parse interval itself may straddle a tight endpoint.
    with ctx.workprec(256):
        return ball.real.contains(arb(str(mp.re(value)))) and ball.imag.contains(arb(str(mp.im(value))))


def test_right_half_plane_square_root_is_sharp_and_contains_independent_values():
    with ctx.workprec(128), mp.workdps(70):
        z = acb(arb(2, 1), arb(0, "1e-12"))
        root = right_sqrt(z)
        assert float(abs(root.imag).upper()) < 1e-12
        for real in [1, 1.3, 2, 3]:
            for imag in ["-1e-12", "0", "1e-12"]:
                assert contains_mp(root, mp.sqrt(mp.mpc(real, imag)))
        with pytest.raises(ValueError, match="right half-plane"):
            right_sqrt(acb(arb(0, 1)))


@pytest.mark.parametrize("z", ["0", ".2+.1j", "-.3+.05j"])
def test_entire_stumpff_values_and_derivatives_contain_independent_series(z):
    with ctx.workprec(128), mp.workdps(80):
        value = mp.mpc(complex(z))
        argument = Dual(acb(str(value.real), str(value.imag)), [acb(1)]+[acb(0)]*6)
        c, s = stumpff_dual(argument)
        for result, shift in [(c, 2), (s, 3)]:
            expected = mp.fsum([(-value)**k/mp.factorial(2*k+shift) for k in range(80)])
            derivative = mp.fsum([(-1)**k*k*value**(k-1)/mp.factorial(2*k+shift)
                                 for k in range(1, 80)])
            assert contains_mp(result.v, expected)
            assert contains_mp(result.d[0], derivative)
        with pytest.raises(ValueError, match="Stumpff range"):
            stumpff_dual(Dual(.6))


def independent_complex_projection(chart, parameters, reception_time):
    """Unscaled universal anomaly; no Sundman implementation or interval jets."""
    q = list(map(mp.mpf, parameters))
    rho, eta, wy, x, y, wx = q
    normal, east, north = [list(map(mp.mpf, axis)) for axis in chart.axes]
    observer0, velocity0 = map(lambda a: list(map(mp.mpf, a)), [chart.observer, chart.velocity])
    direction = [n+x*e+y*b for n, e, b in zip(normal, east, north)]
    r0 = [o+d/rho for o, d in zip(observer0, direction)]
    v0 = [o+(eta*d+wx*e+wy*b)/rho for o, d, e, b in zip(velocity0, direction, east, north)]
    dot = lambda a, b: mp.fsum(u*v for u, v in zip(a, b))
    mu, speed = mp.mpf(GM["sun"]), mp.mpf(C_AU_DAY)
    root_mu = mp.sqrt(mu)
    radius = mp.sqrt(dot(r0, r0))
    alpha = 2/radius-dot(v0, v0)/mu
    rv = dot(r0, v0)/root_mu
    reference = -mp.sqrt(dot(direction, direction))/(rho*speed)
    observer = [mp.cos(root_mu*reception_time), mp.sin(root_mu*reception_time), mp.mpf(0)]

    def propagate(chi):
        z = alpha*chi*chi
        # Independent long series avoids cancellation in complex trig formulas.
        c = mp.fsum([(-z)**k/mp.factorial(2*k+2) for k in range(40)])
        s = mp.fsum([(-z)**k/mp.factorial(2*k+3) for k in range(40)])
        elapsed = (rv*chi*chi*c+(1-alpha*radius)*chi**3*s+radius*chi)/root_mu
        r = [(1-chi*chi*c/radius)*a+(elapsed-chi**3*s/root_mu)*b for a, b in zip(r0, v0)]
        return elapsed, [a-b for a, b in zip(r, observer)]

    def equation(chi):
        elapsed, delta = propagate(chi)
        return reference+elapsed-reception_time+mp.sqrt(dot(delta, delta))/speed

    chi = mp.findroot(equation, root_mu*reception_time/radius, tol=mp.mpf("1e-65"))
    _, delta = propagate(chi)
    denominator = dot(delta, normal)
    return [dot(delta, axis)/denominator for axis in [east, north]]


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
def test_whole_cell_complex_gradients_enclose_independent_universal_derivatives(case):
    chart, center, widths, geometry, _ = setup(case)
    span = float(max(abs(geometry.times)))
    # Full cells, including the stress cover which originally failed without
    # analysis partitioning. q scales remain those of the original full cell.
    gradients, record = partitioned_gradients(
        chart, center, widths, span/2, span/2, 1.5, partition_depth=6)
    assert record["connected_to_known_reference_emission_root"]
    for subcell in record["analysis_cells"]:
        assert subcell["contraction_upper"] < 1
        assert subcell["image_radius_upper"] < subcell["anomaly_disk_radius"]
        assert subcell["positive_real_retarded_root_derivative"]
    with ctx.workprec(128), mp.workdps(80):
        q = [mp.mpf(c)+mp.mpf(w)*mp.mpf(".17") for c, w in zip(center, widths)]
        time = mp.mpc(str(.4*span), str(.1*span))
        step = mp.mpf("1e-20")
        for axis in range(6):
            plus, minus = q.copy(), q.copy()
            plus[axis] += step*mp.mpf(widths[axis])
            minus[axis] -= step*mp.mpf(widths[axis])
            a = independent_complex_projection(chart, plus, time)
            b = independent_complex_projection(chart, minus, time)
            for j in range(2):
                derivative = (a[j]-b[j])/(2*step)
                assert contains_mp(gradients[j][axis], derivative)


def test_real_time_gradient_and_reference_branch_and_order_certificate():
    chart, center, widths, _, _ = setup("near")
    gradients, record = root_and_gradients(chart, center, widths/2, 0, 4, rho=1)
    assert all(v.imag.is_zero() for row in gradients for v in row)
    assert record["connected_to_known_reference_emission_root"]
    assert record["positive_real_retarded_root_derivative"]
    orders, error = choose_orders(1, 18., 2., [12., 9.], tolerance=1e-8)
    assert min(orders) >= 2 and error <= 1e-8
    # A time cell not adjoining the reference cannot silently claim a proved
    # physical branch with this certificate alone.
    _, away = root_and_gradients(chart, center, widths/10, 20, .1, rho=1.2)
    assert not away["connected_to_known_reference_emission_root"]
