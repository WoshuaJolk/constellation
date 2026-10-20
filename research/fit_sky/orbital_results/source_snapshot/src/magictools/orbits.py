"""Massless-particle propagation in a prescribed Solar System gravitational field.

States: solar-system barycentric ICRS axes, AU and AU/day; times: MJD TDB.
Massive-body positions are ephemerides, not independently integrated planets.
"""

from pathlib import Path

from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

# DE440 parameters, https://ssd.jpl.nasa.gov/astro_par.html (km^3/s^2).
# Giant-planet system masses are represented by one point each; no Pluto.
GM_KM = {"sun": 132712440041.279419, "mercury": 22031.868551,
         "venus": 324858.592, "earth": 398600.435507, "moon": 4902.800118,
         "mars": 42828.375816, "jupiter": 126712764.1, "saturn": 37940584.8418,
         "uranus": 5794556.4, "neptune": 6836527.10058}
AU_KM = 149597870.7
GM = {key: value * 86400**2 / AU_KM**3 for key, value in GM_KM.items()}
C_AU_DAY = 299792.458 * 86400 / AU_KM


def observatory_positions(field, times, observatory):
    """Earth barycenter plus IERS-rotated site offset in celestial axes.

    The meter-scale GCRS/ICRS origin-axis relativistic corrections are omitted,
    consistent with the Newtonian astrometric direction model. IERS is offline.
    """
    from .io import _fields
    _fields(observatory, ["longitude_deg", "latitude_deg", "height_m"])
    lon, lat, height = [observatory[k] for k in ["longitude_deg", "latitude_deg", "height_m"]]
    if not np.all(np.isfinite([lon, lat, height])) or not -180 <= lon <= 180 or not -90 <= lat <= 90 or not -500 <= height <= 10000:
        raise ValueError("Invalid observatory geodetic coordinates")
    site = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)
    with iers.conf.set_temp("auto_download", False):
        position, _ = site.get_gcrs_posvel(Time(times, format="mjd", scale="tdb"))
    return field.earth_position(times) + position.xyz.to_value(u.au).T


class SolarSystemField:
    """Hermite-interpolated ephemerides, with a bounded offline default.

    `kernel` is 'builtin' or a local JPL SPK file (requires jplephem). Nothing
    downloads implicitly. Interpolation uses both ephemeris positions/velocities.
    The 1900--2100 interval bounds this implementation, not an accuracy promise.
    """

    def __init__(self, start_mjd, end_mjd, kernel="builtin", step_days=0.25):
        if not np.all(np.isfinite([start_mjd, end_mjd, step_days])) or end_mjd <= start_mjd or not 0 < step_days <= 1:
            raise ValueError("Ephemeris interval must be increasing; step_days must be in (0,1]")
        if start_mjd < 15020 or end_mjd > 88069:
            raise ValueError("Supported ephemeris interval is 1900--2100")
        if kernel != "builtin":
            kernel = str(Path(kernel).resolve(strict=True))
        self.kernel = kernel
        self.start, self.end = float(start_mjd), float(end_mjd)
        self.names = tuple(GM)
        self.masses = np.array([GM[name] for name in self.names])
        count = int(np.ceil((end_mjd - start_mjd) / step_days)) + 1
        if count > 100000:
            raise ValueError("Ephemeris grid exceeds 100,000 samples; shorten the interval")
        offsets = np.linspace(0, end_mjd - start_mjd, count)
        times = Time(start_mjd + offsets, format="mjd", scale="tdb")
        positions, velocities = [], []
        for name in self.names:
            p, v = get_body_barycentric_posvel(name, times, ephemeris=kernel)
            positions.append(p.xyz.to_value(u.au).T)
            velocities.append(v.xyz.to_value(u.au / u.day).T)
        self.spline = CubicHermiteSpline(offsets, np.stack(positions, axis=1),
                                        np.stack(velocities, axis=1), extrapolate=False)

    def positions(self, mjd):
        result = self.spline(np.asarray(mjd) - self.start)
        if not np.all(np.isfinite(result)):
            raise ValueError("Requested time lies outside the ephemeris interval")
        return result

    def earth_position(self, mjd):
        return self.positions(mjd)[..., self.names.index("earth"), :]

    def acceleration(self, mjd, position):
        delta = self.positions(mjd) - position
        distance = np.linalg.norm(delta, axis=-1)
        # Do not integrate singularities or pretend point masses resolve encounters.
        if np.any(distance < 1e-5):
            raise ValueError("Trajectory approaches a gravitating point within 1e-5 AU")
        return np.sum(self.masses[:, None] * delta / distance[:, None]**3, axis=0)


class OrbitSolution:
    """Dense forward/backward DOP853 solution; never extrapolates silently."""

    def __init__(self, state, epoch_mjd, start_mjd, end_mjd, acceleration,
                 rtol=1e-11, atol=1e-13, max_step_days=2):
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("State must contain six finite barycentric position/velocity values")
        if not np.all(np.isfinite([epoch_mjd, start_mjd, end_mjd, rtol, atol, max_step_days])) or not start_mjd <= epoch_mjd <= end_mjd:
            raise ValueError("Propagation interval must include the finite state epoch")
        if not 1e-14 <= rtol <= 1e-3 or not 0 < atol <= 1e-3 or max_step_days <= 0:
            raise ValueError("Invalid integration tolerance or step limit")
        self.epoch, self.start, self.end = epoch_mjd, start_mjd, end_mjd
        self.initial = state.copy()
        self.solutions = {}
        self.nfev = 0
        def derivative(offset, value):
            return np.r_[value[3:], acceleration(epoch_mjd + offset, value[:3])]
        for direction, boundary in [(-1, start_mjd), (1, end_mjd)]:
            if boundary == epoch_mjd:
                continue
            result = solve_ivp(derivative, [0, boundary - epoch_mjd], state,
                               method="DOP853", rtol=rtol, atol=atol,
                               max_step=max_step_days, dense_output=True)
            if not result.success:
                raise ValueError(f"Orbit integration failed: {result.message}")
            self.solutions[direction] = result.sol
            self.nfev += result.nfev

    def at(self, mjd):
        times = np.asarray(mjd, dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times < self.start) or np.any(times > self.end):
            raise ValueError("Requested time lies outside the propagated interval")
        flat = times.ravel()
        output = np.tile(self.initial, (len(flat), 1))
        for direction in [-1, 1]:
            select = (flat - self.epoch) * direction > 0
            if np.any(select):
                output[select] = self.solutions[direction](flat[select] - self.epoch).T
        return output.reshape(times.shape + (6,))


def astrometric_direction(orbit, reception_mjd, observer_position, max_iterations=20):
    """Retarded direction in ICRS axes: object at emission, observer at reception.

    Omits relativistic deflection, aberration and atmospheric refraction. This is
    an astrometric direction, not an apparent AltAz coordinate. Supplied observer
    positions may represent an observatory or spacecraft; Earth default is geocenter.
    """
    observer = np.asarray(observer_position, dtype=float)
    if observer.shape != (3,) or not np.all(np.isfinite(observer)):
        raise ValueError("Observer position must contain three finite barycentric AU values")
    emission = float(reception_mjd)
    for iteration in range(max_iterations):
        delta = orbit.at(emission)[:3] - observer
        distance = np.linalg.norm(delta)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Object and observer coincide or have non-finite separation")
        updated = reception_mjd - distance / C_AU_DAY
        if abs(updated - emission) < 1e-10:
            emission = updated
            break
        emission = updated
    else:
        raise ValueError("Light-time iteration did not converge")
    delta = orbit.at(emission)[:3] - observer
    distance = np.linalg.norm(delta)
    return {"ra_deg": float(np.degrees(np.arctan2(delta[1], delta[0])) % 360),
            "dec_deg": float(np.degrees(np.arctan2(delta[2], np.hypot(delta[0], delta[1])))),
            "distance_au": float(distance), "emission_mjd_tdb": float(emission),
            "light_time_days": float(reception_mjd - emission), "iterations": iteration + 1}


def propagate_family(states, epoch_mjd, reception_times, kernel="builtin", light_time_margin_days=10,
                     rtol=1e-11, atol=1e-13, ephemeris_step_days=0.25):
    """Construct a shared force field and one independent solution per state.

    Ten days of lookback supports ~1,700 AU light times; explicitly increase for
    more distant objects. Out-of-interval light times are errors, not extrapolated.
    """
    times = np.asarray(reception_times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("Reception times must be a nonempty finite vector")
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6 or len(states) == 0 or not np.all(np.isfinite(states)):
        raise ValueError("States must have shape (family_size, 6)")
    if not np.isfinite(light_time_margin_days) or light_time_margin_days <= 0:
        raise ValueError("Light-time margin must be finite and positive")
    start = min(float(times.min()) - light_time_margin_days, epoch_mjd)
    end = max(float(times.max()), epoch_mjd)
    field = SolarSystemField(start - 0.01, end + 0.01, kernel, ephemeris_step_days)
    orbits = [OrbitSolution(state, epoch_mjd, start, end, field.acceleration, rtol, atol) for state in states]
    return field, orbits
