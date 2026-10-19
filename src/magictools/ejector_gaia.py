"""Explicit Gaia DR3 likelihood-to-joint-draw adapter for find_ejector.

Distance prior and missing-RV prior are assumptions, never measurements.
All catalogue rows receive an inclusion/exclusion ledger entry.
"""
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.integrate import cumulative_trapezoid

from .io import _fields, _provenance, read_json
from .stellar_inputs import icrs_to_galactocentric

PARAMETERS = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
REQUIRED = ['source_id', 'ref_epoch'] + PARAMETERS + [f'{p}_error' for p in PARAMETERS] + [f'{a}_{b}_corr' for a,b in combinations(PARAMETERS, 2)]


def read_gaia(path):
    path = Path(path)
    if path.suffix.lower() in {'.vot', '.xml'}:
        root = ET.parse(path).getroot()
        statuses = [x.attrib.get('value') for x in root.iter() if x.tag.split('}')[-1] == 'INFO' and x.attrib.get('name') == 'QUERY_STATUS']
        if not statuses or any(s != 'OK' for s in statuses):
            raise ValueError(f'TAP response missing successful status or truncated/failed: {statuses}')
        table = Table.read(path, format='votable', use_names_over_ids=True)
    else:
        table = Table.read(path)
    names = [name.lower() for name in table.colnames]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous case-insensitive catalogue columns')
    table.rename_columns(table.colnames, names)
    return table


def number(row, key):
    if key not in row.colnames or np.ma.is_masked(row[key]):
        return float('nan')
    try:
        return float(row[key])
    except (ValueError, TypeError):
        return float('nan')


def distance_draws(parallax, error, zero_points, length_pc, quantiles):
    """EDSD distance posterior r² exp(-r/L) times Gaussian parallax likelihood.

    Numerical inverse CDF on a log grid, including dr Jacobian. Explicit finite
    integration limits: 0.1 pc to max(100 kpc, 50 L). Boundary mass is checked.
    """
    if error <= 0 or length_pc <= 0 or not np.isfinite([parallax, error, length_pc]).all():
        raise ValueError('Finite parallax, positive error and distance-prior scale required')
    zero_points, quantiles = np.asarray(zero_points), np.asarray(quantiles)
    if zero_points.shape != quantiles.shape or not np.isfinite(zero_points).all() or np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError('Finite zero points and open-unit quantiles of matching shape required')
    # Add a high-resolution likelihood grid for high-S/N parallaxes.
    upper = max(100000., 50 * length_pc)
    grid = np.geomspace(.1, upper, 4096)
    corrected = parallax - zero_points
    plx_grid = np.linspace(max(.001, float(corrected.min() - 12 * error)), float(max(.002, corrected.max() + 12 * error)), 2048)
    grid = np.unique(np.r_[grid, 1000 / plx_grid])
    grid = grid[(grid >= .1) & (grid <= upper)]
    logp = 2 * np.log(grid)[None, :] - grid[None, :] / length_pc - .5 * ((parallax - zero_points[:, None] - 1000 / grid[None, :]) / error)**2
    density = np.exp(logp - logp.max(axis=1, keepdims=True))
    cdf = cumulative_trapezoid(density, grid, axis=1, initial=0)
    cdf /= cdf[:, -1, None]
    if np.any(cdf[:, 1] > 1e-7) or np.any(1 - cdf[:, -2] > 1e-7):
        raise ValueError('Distance posterior touches integration boundary')
    return np.array([np.interp(q, c, grid) for q,c in zip(quantiles, cdf)])


def gaia_joint_draws(table, n_samples, seed, frame, priors):
    _fields(priors, ['distance_length_pc', 'missing_rv_mean_kms', 'missing_rv_sigma_kms',
                     'parallax_zero_point_mas', 'shared_zero_point_sigma_mas'])
    if not all(np.isfinite(v) for v in priors.values()) or priors['distance_length_pc'] <= 0 or priors['missing_rv_sigma_kms'] <= 0 or priors['shared_zero_point_sigma_mas'] < 0:
        raise ValueError('Invalid explicit distance/RV/zero-point priors')
    if not isinstance(n_samples, int) or not 16 <= n_samples <= 4096:
        raise ValueError('n_samples must be 16--4096')
    if 'source_id' not in table.colnames:
        raise ValueError('Catalogue lacks source_id')
    ids = [str(row['source_id']) for row in table]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source IDs')
    rng = np.random.default_rng(seed)
    zero_points = rng.normal(priors['parallax_zero_point_mas'], priors['shared_zero_point_sigma_mas'], n_samples)
    clouds, included, strata, ledger = [], [], [], []
    missing = sorted(set(REQUIRED) - set(table.colnames))
    for row, source_id in zip(table, ids):
        entry = dict(source_id=source_id, included=False)
        ledger.append(entry)
        if missing:
            entry['reason'] = 'missing_required_columns: ' + ','.join(missing)
            continue
        values = np.array([number(row, key) for key in PARAMETERS])
        errors = np.array([number(row, key + '_error') for key in PARAMETERS])
        corr = np.eye(5)
        for i,j in combinations(range(5), 2):
            corr[i,j] = corr[j,i] = number(row, PARAMETERS[i] + '_' + PARAMETERS[j] + '_corr')
        if not np.isfinite(np.r_[values, errors, corr.ravel()]).all() or np.any(errors <= 0):
            entry['reason'] = 'missing_or_invalid_astrometric_likelihood'
            continue
        if number(row, 'ref_epoch') != 2016.0:
            entry['reason'] = 'epoch_not_Gaia_DR3_2016'
            continue
        if not 0 <= values[0] < 360 or not -90 <= values[1] <= 90 or np.any(np.abs(corr) > 1) or np.linalg.eigvalsh(corr).min() < -1e-10:
            entry['reason'] = 'invalid_coordinates_or_correlation_matrix'
            continue
        covariance = corr * errors[:, None] * errors[None, :]
        remaining = [0, 1, 3, 4]
        cross = covariance[remaining, 2] / covariance[2, 2]
        conditional = covariance[np.ix_(remaining, remaining)] - np.outer(covariance[remaining, 2], covariance[2, remaining]) / covariance[2, 2]
        eig, vectors = np.linalg.eigh(conditional)
        transform = vectors * np.sqrt(np.maximum(0, eig))
        try:
            distance = distance_draws(values[2], errors[2], zero_points, priors['distance_length_pc'], rng.uniform(np.finfo(float).eps, 1, n_samples))
        except ValueError as error:
            entry['reason'] = str(error)
            continue
        delta = rng.normal(size=(n_samples, 4)) @ transform.T + (1000 / distance + zero_points - values[2])[:, None] * cross
        sky = SkyCoord(ra=values[0] * u.deg, dec=values[1] * u.deg).spherical_offsets_by(delta[:, 0]*u.mas, delta[:, 1]*u.mas)
        rv, rv_error = number(row, 'radial_velocity'), number(row, 'radial_velocity_error')
        measured = np.isfinite([rv, rv_error]).all() and 0 < rv_error <= 50 and abs(rv) <= 700 and number(row, 'rv_nb_transits') >= 5
        radial = rng.normal(rv if measured else priors['missing_rv_mean_kms'], rv_error if measured else priors['missing_rv_sigma_kms'], n_samples)
        cloud = np.column_stack([sky.ra.deg, sky.dec.deg, distance, values[3] + delta[:, 2], values[4] + delta[:, 3], radial])
        clouds.append(cloud)
        included.append(source_id)
        # Deliberately coarse diagnostic null stratification. Not a physical halo model.
        strata.append(('measured' if measured else 'prior') + ('_north' if values[1] >= 0 else '_south'))
        entry.update(included=True, radial_velocity='measurement' if measured else 'explicit_prior',
                     distance_percentiles_pc=np.quantile(distance, [.16, .5, .84]).tolist())
    if clouds:
        icrs = np.stack(clouds, axis=1)
        states = icrs_to_galactocentric(icrs.reshape(-1, 6), frame).reshape(icrs.shape)
    else:
        states = np.empty((n_samples, 0, 6))
    return states, dict(source_ids=included, control_strata=strata, ledger=ledger,
        shared_zero_point_draws_mas=zero_points.tolist(), priors=priors, frame=frame,
        input_rows=len(table), included_rows=len(included), excluded_rows=len(table)-len(included),
        caveats=['EDSD is a chosen distance prior, not a halo-population inference',
                 'RV prior is heliocentric Gaussian; unmeasured RV remains unmeasured',
                 'Global zero-point approximation is not the source-dependent Gaia DR3 calibration',
                 'Gaussian single-source astrometry omits unresolved binaries, heavy tails and spatial systematics',
                 'Fixed solar frame; rerun scenarios to assess model sensitivity'])


def run_prepare(manifest_path):
    path = Path(manifest_path).resolve()
    spec = read_json(path)
    _fields(spec, ['schema_version', 'input', 'output', 'n_samples', 'seed', 'frame', 'priors'])
    if spec['schema_version'] != 1:
        raise ValueError('Preparation schema_version must be 1')
    source, output = (path.parent / spec['input']).resolve(), (path.parent / spec['output']).resolve()
    audit_path = output.with_suffix('.audit.json')
    if output.exists() or audit_path.exists():
        raise ValueError('Preparation outputs already exist')
    states, audit = gaia_joint_draws(read_gaia(source), spec['n_samples'], spec['seed'], spec['frame'], spec['priors'])
    audit['provenance'] = _provenance([path, source])
    import json
    with audit_path.open('x') as stream:
        stream.write(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    with output.open('xb') as stream:
        np.savez_compressed(stream, states_pc_kms=states, source_ids=np.asarray(audit['source_ids'], dtype=str), control_strata=np.asarray(audit['control_strata'], dtype=str))
    return audit
