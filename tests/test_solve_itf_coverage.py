import json
from types import SimpleNamespace

import numpy as np
from astropy.coordinates import EarthLocation

from magictools.itf.catalogue import OBS_DTYPE, TRACK_DTYPE
from magictools.itf.coverage import observing_coverage, site_longitudes, write_coverage


def test_midnight_does_not_split_local_night_but_noon_does():
    # Greenwich evening and next morning are one night.
    assert observing_coverage([60000.9, 60001.1], 0)['nights'] == 1
    assert observing_coverage([60001.49, 60001.51], 0)['nights'] == 2
    # Hawaii: observations straddling UTC noon remain in one local night.
    assert observing_coverage([60000.49, 60000.51], -155)['nights'] == 1
    # East longitude: local noon crosses before UTC noon.
    assert observing_coverage([60000.16, 60000.18], 120)['nights'] == 2


def test_longitude_wrap_and_same_night_at_multiple_stations():
    times = [60000.99, 60001.1, 60001.2, 60002.1]
    # Chile and Hawaii: first three observations share a local evening date.
    assert observing_coverage(times, [-70, -70, -155, -70])['nights'] == 2
    assert observing_coverage(times, [290, 290, 205, 290])['nights'] == 2
    assert observing_coverage([60000.7, 60000.7], [-180, 180])['nights'] == 1


def test_season_proxy_unchanged_and_unknown_nights_explicit():
    times = np.array([60000.1, 60000.8, 60001.01, 60120.1, 60240.5, 60605.1])
    assert observing_coverage(times, -155)['oppositions_estimate'] == 3
    assert observing_coverage([], []) == {'nights': 0, 'oppositions_estimate': 0}
    assert observing_coverage([60000.1, 60001.1], [0, np.nan]) == {
        'nights': None, 'oppositions_estimate': 1}
    assert site_longitudes({b'500': EarthLocation.from_geocentric(0, 0, 0, unit='m')}) == {}


def test_report_backfill_uses_each_observation_time_and_station(tmp_path):
    obs = np.zeros(4, dtype=OBS_DTYPE)
    obs['t'] = [60000.99, 60001.1, 60001.2, 60002.1]
    tracks = np.zeros(2, dtype=TRACK_DTYPE)
    tracks['site'], tracks['start'], tracks['count'] = [b'W84', b'F51'], [0, 2], [2, 2]
    catalogue = SimpleNamespace(obs=obs, tracks=tracks, manifest={'source_sha256': 'test'})
    path = tmp_path/'report.json'
    original = json.dumps({'input': catalogue.manifest, 'candidates': [{'tracklet_indices': [0, 1]}]})
    path.write_text(original)
    sites = {b'W84': EarthLocation.from_geodetic(-70, -30),
             b'F51': EarthLocation.from_geodetic(-155, 20)}
    result = write_coverage(path, catalogue, sites)
    assert result['chains']['0,1'] == {'nights': 2, 'oppositions_estimate': 1}
    assert result['schema'] == 2
    assert json.loads(path.with_name('observing_coverage.json').read_text()) == result
    assert path.read_text() == original  # Preserve scientific report provenance.
