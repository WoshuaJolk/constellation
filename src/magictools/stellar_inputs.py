"""Sampling supplied 6D stellar posteriors; no implicit parallax or RV priors."""

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianDifferential, Galactocentric, SkyCoord

from .io import _fields


def icrs_to_galactocentric(samples, frame_spec):
    """Columns: RA deg, Dec deg, distance pc, pm_ra_cosdec/pm_dec mas/yr, RV km/s."""
    _fields(frame_spec, ["galcen_distance_pc", "z_sun_pc", "galcen_v_sun_kms"])
    distance, height = frame_spec["galcen_distance_pc"], frame_spec["z_sun_pc"]
    velocity = np.asarray(frame_spec["galcen_v_sun_kms"], dtype=float)
    if not np.all(np.isfinite([distance, height])) or distance <= 0 or abs(height) >= distance or velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("Invalid explicit Galactocentric frame parameters")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("ICRS samples must be a nonempty finite N by 6 matrix")
    ra, dec, dist, pmra, pmdec, rv = values.T
    if np.any((ra < 0) | (ra >= 360) | (dec < -90) | (dec > 90) | (dist <= 0)):
        raise ValueError("ICRS samples require valid RA/Dec and positive posterior distances")
    # Explicit center orientation and roll avoid dependence on Astropy defaults.
    frame = Galactocentric(galcen_distance=distance * u.pc, z_sun=height * u.pc,
        galcen_v_sun=CartesianDifferential(velocity * u.km / u.s),
        galcen_coord=SkyCoord(ra=266.4051 * u.deg, dec=-28.936175 * u.deg), roll=0 * u.deg)
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=dist * u.pc,
                   pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                   radial_velocity=rv * u.km / u.s, frame="icrs").transform_to(frame)
    return np.column_stack([sky.cartesian.xyz.to_value(u.pc).T, sky.velocity.d_xyz.to_value(u.km / u.s).T])


def sample_catalogue(rows, n_samples, rng, frame_spec=None):
    """Independent stars, retaining each star's supplied position/velocity covariance.

    Per-star posterior sample rows are resampled as indivisible six-vectors.
    Cross-star systematics/covariance are not modeled by this input format.
    """
    clouds = []
    for row in rows:
        _fields(row, ["id"], ["state_pc_kms", "covariance", "samples_pc_kms", "icrs_samples", "control_stratum", "label"])
        choices = set(row) & {"state_pc_kms", "samples_pc_kms", "icrs_samples"}
        if len(choices) != 1:
            raise ValueError("Each object requires exactly one state or posterior-sample format")
        choice = next(iter(choices))
        if choice == "state_pc_kms":
            state = np.asarray(row[choice], dtype=float)
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise ValueError("state_pc_kms must be a finite six-vector")
            covariance = np.asarray(row.get("covariance", np.zeros((6, 6))), dtype=float)
            if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
                raise ValueError("Covariance must be a finite 6 by 6 matrix")
            tolerance = 1e-12 * max(1, np.max(np.abs(covariance)))
            if not np.allclose(covariance, covariance.T, atol=tolerance, rtol=1e-12):
                raise ValueError("Covariance must be symmetric")
            eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
            if eigenvalues.min() < -tolerance:
                raise ValueError("Covariance must be positive semidefinite")
            transform = vectors * np.sqrt(np.maximum(0, eigenvalues))
            cloud = state + rng.normal(size=(n_samples, 6)) @ transform.T
        else:
            if "covariance" in row:
                raise ValueError("A covariance cannot be combined with posterior samples")
            values = np.asarray(row[choice], dtype=float)
            if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError("Posterior samples must be a nonempty finite N by 6 matrix")
            if choice == "icrs_samples":
                if frame_spec is None:
                    raise ValueError("ICRS samples require explicit galactocentric_frame parameters")
                values = icrs_to_galactocentric(values, frame_spec)
            cloud = values[rng.integers(0, len(values), size=n_samples)]
        clouds.append(cloud)
    return np.stack(clouds, axis=1)
