"""Local mean-solar observing nights and a UTC observing-season proxy."""
import json
from pathlib import Path

import numpy as np

NIGHTS_DEFINITION = (
    'Distinct local evening dates: noon-to-noon in local mean solar time at each '
    'observatory longitude (east-positive, wrapped to [-180, 180) degrees). '
    'The same local date at multiple stations counts once. Unknown or non-terrestrial '
    'stations make the night count unavailable; no UTC fallback.'
)


def site_longitudes(sites):
    """Exclude geocentric placeholders, which have no terrestrial observing night."""
    return {code: float(location.lon.deg) for code, location in sites.items()
            if np.linalg.norm([location.x.to_value('m'), location.y.to_value('m'),
                               location.z.to_value('m')]) > 1.}


def observing_coverage(times, longitudes, season_gap_days=120):
    times = np.asarray(times, dtype=float)
    longitude = np.broadcast_to(np.asarray(longitudes, dtype=float), times.shape)
    utc_dates = np.unique(np.floor(times).astype(np.int64))
    # Noon starts the date-labelled observing night; midnight never splits it.
    nights = (int(len(np.unique(np.floor(times + ((longitude+180.) % 360.-180.)/360. - .5))))
              if np.isfinite(longitude).all() else None)
    return dict(nights=nights, oppositions_estimate=int(bool(len(utc_dates))) +
                int(np.sum(np.diff(utc_dates) >= season_gap_days)))


def write_coverage(report_path, catalogue, sites):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report['input'] != catalogue.manifest:
        raise ValueError('Report and coverage cache differ')
    longitudes = site_longitudes(sites)
    per_track = {}
    chains = {}
    for row in report['candidates']:
        for index in row['tracklet_indices']:
            if index not in per_track:
                track = catalogue.tracks[index]
                times = catalogue.obs['t'][track['start']:track['start'] + track['count']]
                per_track[index] = (times, np.full(len(times), longitudes.get(track['site'], np.nan)))
        chains[','.join(map(str, row['tracklet_indices']))] = observing_coverage(
            np.concatenate([per_track[index][0] for index in row['tracklet_indices']]),
            np.concatenate([per_track[index][1] for index in row['tracklet_indices']]))
    product = dict(schema=2, source_sha256=report['input']['source_sha256'], season_gap_days=120,
                   nights_definition=NIGHTS_DEFINITION,
                   observatory_longitudes_deg={code.decode(): longitude for code, longitude in longitudes.items()},
                   oppositions_definition='Estimated observed seasons separated by gaps of at least 120 UTC days; not geometric opposition events',
                   chains=chains)
    output = report_path.with_name('observing_coverage.json')
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(product, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(output)
    return product
