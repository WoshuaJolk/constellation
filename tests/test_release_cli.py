"""Release acceptance: real CLI calls, independent baselines, explicit scope.

Set CONSTELLATION_RELEASE_REPORT to save measured results as JSON. Network
protocol checks use a local HTTP archive; public service availability is separate.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(sys.executable).with_name("constellation")


def invoke(folder, command, *args, code=0):
    result = subprocess.run([str(CLI), command, *map(str, args)], cwd=folder,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == code, result.stderr + result.stdout
    return json.loads(result.stdout)


def save(folder, name, value):
    path = folder / name
    path.write_text(json.dumps(value, allow_nan=False))
    return path


@pytest.fixture(scope="module")
def evidence():
    rows = {}
    def record(tool, baseline, improvement, **metrics):
        rows[tool] = {"baseline": baseline, "demonstrated_value": improvement, "metrics": metrics,
                      "status": "passed", "scope": "Declared control only; not a survey-wide scientific guarantee"}
    yield record
    destination = os.environ.get("CONSTELLATION_RELEASE_REPORT")
    if destination:
        from magictools.release import READY
        digest = hashlib.sha256()
        for root in [ROOT / "src/magictools", ROOT / "research"]:
            for path in sorted(root.rglob('*.py')):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
        for path in (Path(__file__), ROOT / 'examples/matching.json', ROOT / 'research/ejector_fit/data/hip1997_triple.json'):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        report = {"schema_version": 1, "source_sha256": digest.hexdigest(),
                  "complete": set(rows) == set(READY), "expected_tools": sorted(READY),
                  "tested_tools": sorted(rows), "results": rows,
                  "network": "Local HTTP archive controls; no assumption about public archive uptime",
                  "limits": "Fixed controlled cases and stated baselines. Passing is not a discovery, global completeness, or calibrated real-survey false-positive rate."}
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


@pytest.fixture
def images(tmp_path):
    # Independently generate pixel-integrated sources with Gaussian CDFs.
    yy, xx = np.mgrid[:41, :41]
    rng = np.random.default_rng(71)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [21, 21]
    wcs.wcs.crval = [150, 2]
    wcs.wcs.cdelt = [-1/3600, 1/3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    exposures, track, wrong, peaks = [], [], [], []
    for i in range(8):
        x, y, sigma = 8 + 3*i, 17 + .3*i, 1.2
        psf = (ndtr((xx + .5 - x)/sigma) - ndtr((xx - .5 - x)/sigma)) * (ndtr((yy + .5 - y)/sigma) - ndtr((yy - .5 - y)/sigma))
        data = 40 + 20 * psf + rng.normal(0, 2, xx.shape)
        name = f'e{i}'
        fits.writeto(tmp_path / f'{name}.fits', data, wcs.to_header())
        exposures.append(dict(id=name, image=f'{name}.fits', variance=4, psf_sigma_pixels=sigma))
        track.append(dict(exposure_id=name, x=x, y=y))
        wrong.append(dict(exposure_id=name, x=x, y=y+12))
        peaks.append(2 / psf[round(y), round(x)])
    manifest = dict(schema_version=1, exposures=exposures,
                    trajectories=[dict(id='true', positions=track), dict(id='wrong', positions=wrong)])
    return tmp_path, save(tmp_path, 'tracks.json', manifest), manifest, peaks


def test_force_track_and_dig_floor_improve_on_single_pixel_and_single_visit(images, evidence):
    folder, path, _, peaks = images
    photo = invoke(folder, 'force_track', path)
    measured = next(t for t in photo['trajectories'] if t['id'] == 'true')['measurements']
    errors = [m['flux_error'] for m in measured]
    assert np.mean(errors) < .6 * np.mean(peaks)
    assert abs(np.mean([m['flux'] for m in measured]) - 20) < 3 * np.mean(errors) / np.sqrt(8)
    evidence('force_track', 'Single central-pixel flux estimate, with the true background and PSF known',
             'Lower flux uncertainty by using the full PSF while fitting a background',
             mean_flux_error=float(np.mean(errors)), single_pixel_error=float(np.mean(peaks)))
    search = invoke(folder, 'dig_floor', path)
    top = search['trajectories'][0]
    assert top['id'] == 'true' and top['joint']['passes_gaussian_threshold']
    best_single = max(m['snr'] for m in measured)
    assert best_single < 5 < top['joint']['snr']
    evidence('dig_floor', 'A 5-sigma test on each exposure separately',
             'Recovers the correct supplied trajectory by combining subthreshold exposures',
             best_single_snr=best_single, joint_snr=top['joint']['snr'], ranked_first=top['id'])


def test_judge_line_rejects_a_high_scoring_one_visit_impostor(images, evidence):
    folder, path, manifest, _ = images
    yy, xx = np.mgrid[:41, :41]
    for i, exposure in enumerate(manifest['exposures']):
        p = manifest['trajectories'][0]['positions'][i]
        psf = (ndtr((xx+.5-p['x'])/1.2)-ndtr((xx-.5-p['x'])/1.2)) * (ndtr((yy+.5-p['y'])/1.2)-ndtr((yy-.5-p['y'])/1.2))
        with fits.open(folder / exposure['image'], mode='update') as h:
            h[0].data[:] = 40 + (400 if i == 0 else 0)*psf
    search = invoke(folder, 'dig_floor', path)
    candidate = search['trajectories'][0]
    assert candidate['joint']['passes_gaussian_threshold']
    judge = save(folder, 'judge.json', dict(schema_version=1, photometry_manifest='tracks.json', candidate_id='true',
                                          times_mjd_tdb={f'e{i}':60000+i/48 for i in range(8)}))
    result = invoke(folder, 'judge_line', judge)
    assert 'jackknife_below_requested_snr' in result['flags']
    assert result['combined_false_positive_probability'] is None
    evidence('judge_line', 'Accepting a high joint score alone', 'Flags a candidate driven by one exposure',
             joint_snr=candidate['joint']['snr'], flags=result['flags'])


def test_inject_tracks_measures_recovery_without_mutating_pixels(images, evidence):
    folder, _, manifest, _ = images
    for e in manifest['exposures']:
        with fits.open(folder / e['image'], mode='update') as h:
            h[0].data[:] = 40
    before = {e['image']: hashlib.sha256((folder/e['image']).read_bytes()).hexdigest() for e in manifest['exposures']}
    path = save(folder, 'inject.json', dict(schema_version=1, photometry_manifest='tracks.json',
                injection_trajectory_id='true', reference_fluxes=[0,100], offsets_arcsec=[[0,0],[0,5]]))
    result = invoke(folder, 'inject_tracks', path)
    fractions = [s['new_recovery_fraction'] for s in result['summary']]
    assert fractions == [0, 1]
    assert before == {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in before}
    evidence('inject_tracks', 'Assuming a nominal sensitivity implies track recovery',
             'Actually reruns the search on zero-flux and bright injections; preserves input pixels',
             new_recovery_fractions=fractions, unchanged_images=len(before))


def test_map_cover_eliminates_masked_area_false_coverage(tmp_path, evidence):
    from magictools.orbit_demo import create_orbit_demo
    root = tmp_path / 'orbit-control'
    create_orbit_demo(root)
    with fits.open(root / 'frame-0.fits', mode='update') as h:
        h['MASK'].data[:] = 1
    result = invoke(tmp_path, 'map_cover', root/'orbits.json')
    first = result['exposures'][0]
    naive = sum(m['on_chip'] for m in first['members'])
    assert naive > 0 and first['weighted_usable_fraction'] == 0
    assert all(not m['usable'] for m in first['members'])
    assert result['planning_manifest']['non_detections'] == []
    evidence('map_cover', 'Counting a predicted position inside image bounds as usable coverage',
             'Rejects masked detector area and does not invent non-detections',
             on_chip_predictions=naive, usable_fraction=first['weighted_usable_fraction'])


def test_weigh_stamp_separates_trail_from_cosmic_ray(tmp_path, evidence):
    yy, xx = np.mgrid[:25, :25]
    def psf(x,y):
        return (ndtr((xx+.5-x)/1.1)-ndtr((xx-.5-x)/1.1))*(ndtr((yy+.5-y)/1.1)-ndtr((yy-.5-y)/1.1))
    # Independent midpoint integration of a uniform linear trail.
    trail = sum(psf(12+8*t,12+2*t) for t in (np.arange(512)+.5)/512-.5)/512
    spike = np.zeros_like(trail); spike[12,12] = 1
    preferred = []
    for name, image in [('trail',trail),('spike',spike)]:
        fits.writeto(tmp_path/f'{name}.fits',40+400*image)
        path=save(tmp_path,f'{name}.json',dict(schema_version=1,image=f'{name}.fits',variance=4,
            x=12,y=12,psf_sigma_pixels=1.1,trail_dx=8,trail_dy=2,flux_prior_sigma=500))
        preferred.append(invoke(tmp_path,'weigh_stamp',path)['preferred_model'])
    assert preferred == ['moving_source','cosmic_ray_spike']
    evidence('weigh_stamp','Equal integrated positive flux for a trail and a spike',
             'Distinguishes their pixel morphologies under the supplied model family',preferred_models=preferred)


def test_learn_null_changes_weights_and_selects_informative_followup(tmp_path,evidence):
    path=save(tmp_path,'null.json',dict(schema_version=1,
        hypotheses=[dict(id='a',prior_weight=1),dict(id='b',prior_weight=1)],
        non_detections=[dict(id='past',detection_probability=[.9,.1])],
        candidate_exposures=[dict(id='uninformative',detection_probability=[.5,.5]),dict(id='informative',detection_probability=[0,1])]))
    result=invoke(tmp_path,'learn_null',path)
    assert result['posterior_weights'] == pytest.approx([.1,.9])
    first,last=result['ranked_exposures']
    assert first['id']=='informative' and first['expected_information_nats'] > .3
    assert last['expected_information_nats'] == pytest.approx(0)
    evidence('learn_null','Uniform hypothesis weights and choosing the first listed follow-up',
             'Uses supplied detection efficiencies to update probabilities and select information',
             posterior=result['posterior_weights'],first_followup=first['id'],information_nats=first['expected_information_nats'])


def test_map_complete_preserves_shared_exposure_dependence(tmp_path,evidence):
    value=dict(schema_version=1,exposures=[dict(id='shared',observation_id='same',surveys=['a','b'],mjd_utc=60000)],
        objects=[dict(id='object',weight=1,detection_probability=[.5])],
        discovery_rules=[dict(id=s,surveys=[s],min_visits_per_night=1,min_nights=1) for s in ['a','b']])
    result=invoke(tmp_path,'map_complete',save(tmp_path,'selection.json',value))
    assert result['population_probabilities']['union']==pytest.approx(.5)
    evidence('map_complete','Independent-survey union calculation gives 0.75 for one shared exposure',
             'Returns the correct 0.5 union by retaining shared-exposure dependence',
             naive_union=.75,correct_union=result['population_probabilities']['union'])


def test_inject_all_extends_selection_beyond_enumeration_budget(tmp_path,evidence):
    n,p=24,.05
    value=dict(schema_version=1,exposures=[dict(id=str(i),observation_id=str(i),surveys=['a'],mjd_utc=60000+i) for i in range(n)],
        objects=[dict(id='object',weight=1,detection_probability=[p]*n)],
        discovery_rules=[dict(id='any',surveys=['a'],min_visits_per_night=1,min_nights=1)],draws_per_object=40000,seed=123,confidence=.99)
    path=save(tmp_path,'large-selection.json',value)
    exact=save(tmp_path,'exact-selection.json',{k:v for k,v in value.items() if k not in {'draws_per_object','seed'}})
    failed=subprocess.run([str(CLI),'map_complete',str(exact)],cwd=tmp_path,capture_output=True,text=True)
    assert failed.returncode == 2 and '20-exposure' in failed.stderr
    result=invoke(tmp_path,'inject_all',path)
    truth=1-(1-p)**n
    assert abs(result['population_probabilities']['union']-truth)<.01
    assert result['union_mc_interval'][0]<truth<result['union_mc_interval'][1]
    evidence('inject_all','Exact enumeration refuses 24 exposures under its declared resource limit',
             'Computes a selection estimate with Monte Carlo uncertainty, checked against an analytic answer',
             exposures=n,analytic_probability=truth,estimate=result['population_probabilities']['union'],interval=result['union_mc_interval'])


def test_assignment_and_certificate_improve_on_greedy_and_trust(tmp_path,evidence):
    value=json.loads((ROOT/'examples/matching.json').read_text())
    result=invoke(tmp_path,'match_all',save(tmp_path,'matching.json',value))
    assert result['objective']==12 and result['status']=='optimal'
    evidence('match_all','Greedy highest-score candidate gives objective 10',
             'Finds the globally optimal compatible assignment for the supplied candidates',greedy=10,optimal=result['objective'])
    report=save(tmp_path,'matched.json',result)
    assert invoke(tmp_path,'verify_match',report)['valid']
    result['objective']+=1
    save(tmp_path,'matched.json',result)
    bad=subprocess.run([str(CLI),'verify_match',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert bad.returncode==2
    evidence('verify_match','Trusting a solver report without checking its certificate',
             'Accepts the valid result and rejects a tampered objective',valid_accepted=True,tampered_rejected=True)


def test_prove_floor_improves_on_known_template_oracle_bound(tmp_path,evidence):
    path=save(tmp_path,'floor.json',dict(schema_version=1,gram_matrix=np.eye(20).tolist(),prior_weights=[1]*20,
             flux=1,alpha=.01,calibration_samples=1024,evaluation_samples=1024,seed=4))
    result=invoke(tmp_path,'prove_floor',path)
    bound=result['bounds']['best_power_upper_bound']
    oracle=result['bounds']['oracle_power_upper_bound']
    expected=.01+np.sqrt(.01*.99*np.expm1(1)/20)
    assert bound==pytest.approx(expected) and bound<oracle
    assert result['detector']['null_rejection_probability_upper_bound']<=.01
    evidence('prove_floor','Known-template oracle bound ignores uncertainty over 20 templates',
             'Produces a tighter prior-average power bound under the stated Gaussian model',
             oracle_upper_bound=oracle,mixture_upper_bound=bound,analytic_check=float(expected))


def test_kepler_propagation_beats_straight_line_prediction(tmp_path,evidence):
    from magictools.orbits import GM
    radius,time=3.,100.
    speed=np.sqrt(GM['sun']/radius)
    value=dict(schema_version=1,states=[dict(id='circular',state=[radius,0,0,0,speed,0])],times_days=[time])
    result=invoke(tmp_path,'kepler-propagate',save(tmp_path,'orbit.json',value))
    assert result['ok']
    report=json.loads(Path(result['artifacts']['report.json']).read_text())
    angle=speed*time/radius
    truth=np.array([radius*np.cos(angle),radius*np.sin(angle),0])
    actual=np.array(report['states'])[0,0,:3]
    linear=np.array([radius,speed*time,0])
    error=float(np.linalg.norm(actual-truth));baseline=float(np.linalg.norm(linear-truth))
    assert error<1e-10 and baseline>.1
    evidence('kepler-propagate','Constant-velocity extrapolation over 100 days',
             'Matches the analytic two-body circular solution',linear_error_au=baseline,propagated_error_au=error)


def test_ejector_fit_recovers_separate_launch_times(tmp_path,evidence):
    spec=json.loads((ROOT/'research/ejector_fit/data/hip1997_triple.json').read_text())
    spec['potential']={'components':[]}
    p,v=np.array([500.,150,100]),np.array([3.,10,0])
    def observation(name,position,velocity):
        d=np.linalg.norm(position);ra=np.arctan2(position[1],position[0]);dec=np.arcsin(position[2]/d)
        east=np.array([-np.sin(ra),np.cos(ra),0]);north=np.array([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)])
        return dict(id=name,ra_deg=float(np.rad2deg(ra)%360),dec_deg=float(np.rad2deg(dec)),
            mean=[1000/d,np.dot(velocity,east)/(4.740470463533349*d/1000),np.dot(velocity,north)/(4.740470463533349*d/1000),np.dot(velocity,position)/d],
            covariance=(np.diag([.01,.02,.02,.5])**2).tolist())
    spec['objects']=[observation('source',p,v)]
    for name,age,speed in [('a',1.3,[50.,40,10]),('b',3.1,[-40.,30,20])]:
        speed=np.array(speed)
        spec['objects'].append(observation(name,p+(speed-v)*age*1.022712165045695,speed))
    spec['encounters']=[dict(id=name,members=['source',name],lookback_bounds_myr=[.1,5.],radius_pc=0.) for name in ['a','b']]
    spec['fit']['starts_myr']=[[.5,.5],[2.,4.]]
    result=invoke(tmp_path,'ejector_fit',save(tmp_path,'launches.json',spec))
    ages=[e['lookback_myr'] for e in result['best_fit']['events']]
    assert ages==pytest.approx([1.3,3.1],abs=1e-6)
    assert result['best_fit']['chi2']<1e-7 and result['discovery_probability'] is None
    common_age_error=min(max(abs(t-1.3),abs(t-3.1)) for t in np.linspace(.1,5,10001))
    evidence('ejector_fit','Assigning one common launch time to two distinct ejections',
             'Fits distinct encounter times sharing one source state within the declared model',
             recovered_ages_myr=ages,best_single_age_max_error_myr=common_age_error,chi2=result['best_fit']['chi2'])


def test_find_and_check_frames_use_real_http_and_detect_corruption(tmp_path,evidence):
    names=['dataproduct_type','obs_id','obs_publisher_did','access_url','s_ra','s_dec','t_min','t_max','s_region','instrument_name']
    columns=''.join(f'<FIELD name="{n}" datatype="char" arraysize="*"/>' for n in names)
    values=['image','visit','native-frame','https://example.invalid/frame.fits',10,20,58849,58849.01,'CIRCLE ICRS 10 20 .2','DECam']
    row='<TR>'+''.join(f'<TD>{v}</TD>' for v in values)+'</TR>'
    payload=f'<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="OK"/><TABLE>{columns}<DATA><TABLEDATA>{row}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'.encode()
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_GET(self):
            requests.append(self.path)
            self.send_response(503 if self.path.startswith('/down') else 200)
            self.end_headers()
            if not self.path.startswith('/down'): self.wfile.write(payload)
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            self.do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        services={f'vo:{p}':dict(protocol='obscore',endpoint=f'http://127.0.0.1:{server.server_port}/{p}/sync') for p in ['one','two','down']}
        value=dict(mode='field',ra=10,dec=20,radius=.1,start='2020-01-01',end='2020-01-03',providers=list(services),services=services)
        result=invoke(tmp_path,'find_frames',save(tmp_path,'query.json',value),'--progress','none',code=3)
        assert result['status']=='incomplete' and len(result['frames'])==1
        assert len(result['frames'][0]['provenance'])==2
        assert next(s for s in result['surveys'] if s['provider']=='vo:down')['status']=='unavailable'
        assert len(requests)==3
        evidence('find_frames','Concatenating duplicate archive rows and mistaking a failed service for no coverage',
                 'Queries multiple services, deduplicates the native frame, and retains the failed-provider status',
                 raw_matching_rows=2,unique_frames=1,failed_provider_visible=True)
        run=Path(result['execution']['runDirectory'])
        checked=invoke(tmp_path,'check_frames',run,'--checksums')
        assert checked['artifactIntegrity']['verified']
        first=next(c for c in result['checks'] if c['status']=='ok')
        response=run/first['artifactDirectory']/first['requests'][0]['responseFile']
        response.write_text('corrupted')
        corrupted=invoke(tmp_path,'check_frames',run,'--checksums',code=4)
        assert len(corrupted['artifactIntegrity']['failures'])==1
        evidence('check_frames','Trusting stored responses and successful query labels',
                 'Detects a modified response and distinguishes query success from complete coverage',
                 checked_responses=checked['artifactIntegrity']['checked'],corruption_detected=True)
    finally:
        server.shutdown();server.server_close();thread.join(3)


def test_promoted_catalogue_and_installed_entrypoint_match_readme(tmp_path):
    import re
    from importlib.metadata import version
    from magictools.release import READY, EXPERIMENTAL
    names=re.findall(r'^\| \[`([^`]+)`\]',(ROOT/'README.md').read_text(),re.M)
    assert len(names)==len(set(names)) and set(names)==set(READY)
    assert not (set(names)&set(EXPERIMENTAL))
    discovered=invoke(tmp_path,'list','--ready')
    assert set(discovered['result']['tools'])==set(READY)
    assert invoke(tmp_path,'doctor')['result']['ready']
    actual=subprocess.run([str(CLI),'--version'],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert actual.stdout.strip()==version('const-ellation')
