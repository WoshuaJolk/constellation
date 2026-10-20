"""Download a small public ZTF field and test synthetic tracks on real images.

Run after installing magictools:
    python examples/ztf_workflow.py /path/to/new-directory
Network access is explicit. This does not search for or claim a real discovery.
"""

import argparse
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import numpy as np

from magictools.coverage import run_cover
from magictools.injections import run_injections
from magictools.io import run_photometry
from magictools.vetting import run_judge
from magictools.ztf import run_ztf_fetch, run_ztf_search


def run(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = directory / name
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        return path
    query = dict(schema_version=1, regions=[dict(ra_deg=180, dec_deg=30, width_deg=.01)],
                 start_mjd_utc=58511.38, end_mjd_utc=58511.41, filtercode='zr', max_exposures=20)
    index = run_ztf_search(save('query.json', query))
    save('index.json', index)
    if not index['exposures']:
        raise ValueError('The archive returned no exposures for this example')
    fetch = dict(schema_version=1, index='index.json', directory='download', reference_mag=25,
                 selections=[dict(pid=row['pid'], ra_deg=180, dec_deg=30, size_arcsec=180) for row in index['exposures']])
    inventory = run_ztf_fetch(save('fetch.json', fetch))
    save('inventory.json', inventory)
    epoch = inventory['exposures'][0]['mjd_tdb']
    earth = get_body_barycentric_posvel('earth', Time(epoch, format='mjd', scale='tdb'))[0].xyz.to_value(u.au)
    # Arbitrary synthetic barycentric hypotheses toward the downloaded field.
    ra, dec = np.radians([180, 30])
    position = earth + 3 * np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    orbits = [dict(id=name, state_au_au_per_day=np.r_[position, [0, speed, 0]].tolist(), prior_weight=1, reference_flux=100)
              for name, speed in [('injection-track', .01), ('slower', 0), ('faster', .02)]]
    cover = run_cover(save('cover-input.json', dict(schema_version=1, epoch_mjd_tdb=epoch, orbits=orbits, inventory='inventory.json')))
    save('cover.json', cover)
    save('tracks.json', cover['photometry_manifest'])
    save('planning.json', cover['planning_manifest'])
    save('force-track.json', run_photometry(directory/'tracks.json'))
    save('dig-floor.json', run_photometry(directory/'tracks.json', joint=True))
    judge = dict(schema_version=1, photometry_manifest='tracks.json', candidate_id='injection-track',
                 times_mjd_tdb={e['id']: e['mjd_tdb'] for e in inventory['exposures']})
    save('judge.json', run_judge(save('judge-input.json', judge)))
    experiment = dict(schema_version=1, photometry_manifest='tracks.json', injection_trajectory_id='injection-track',
                      reference_fluxes=[0, 10, 30, 100, 300], offsets_arcsec=[[x,y] for x in [-40,-20,0,20,40] for y in [-30,0,30]])
    injections = run_injections(save('inject-input.json', experiment))
    save('injections.json', injections)
    summary = dict(real_backgrounds=True, synthetic_injections=True, exposure_ids=[e['id'] for e in inventory['exposures']],
                   recovery=injections['summary'], n_baseline_detections=sum(b['detected_before_injection'] for b in injections['baselines']))
    save('summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(run(parser.parse_args().directory), indent=2))
