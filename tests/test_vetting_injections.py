import hashlib
import json
import subprocess
import sys

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from magictools.image_bank import ImageBank
from magictools.injections import run_injections
from magictools.photometry import gaussian_psf
from magictools.vetting import run_judge


@pytest.fixture
def experiment(tmp_path):
    wcs=WCS(naxis=2); wcs.wcs.crpix=[31,31]; wcs.wcs.crval=[359.999,30]
    wcs.wcs.cdelt=[-1/3600,1/3600]; wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    exposures=[]; moving=[]; wrong=[]
    yy,xx=np.mgrid[:61,:61]
    for i in range(5):
        eid=f'e{i}'; x=18+6*i; y=27+.4*i
        fits.writeto(tmp_path/f'{eid}.fits', 30+100*gaussian_psf(xx,yy,x,y,1.2), wcs.to_header())
        exposures.append(dict(id=eid,image=f'{eid}.fits',variance=4,psf_sigma_pixels=1.2))
        moving.append(dict(exposure_id=eid,x=x,y=y)); wrong.append(dict(exposure_id=eid,x=45-5*i,y=40))
    manifest=dict(schema_version=1,exposures=exposures,trajectories=[dict(id='mover',positions=moving),dict(id='wrong',positions=wrong)])
    path=tmp_path/'tracks.json'; path.write_text(json.dumps(manifest))
    judge=dict(schema_version=1,photometry_manifest='tracks.json',candidate_id='mover',times_mjd_tdb={f'e{i}':60000+i/48 for i in range(5)})
    judge_path=tmp_path/'judge.json'; judge_path.write_text(json.dumps(judge))
    return path,judge_path,manifest


def test_vetting_recovers_mover_and_controls_use_same_exposures(experiment):
    _,path,_=experiment
    result=run_judge(path)
    assert result['flags'] == []
    assert result['combined_false_positive_probability'] is None
    assert len(result['leave_one_out']) == 5
    assert all(row['joint']['snr'] > 10 for row in result['chronological_halves'])
    assert len(result['controls']) == 4
    for control in result['controls']:
        assert control['candidate_snr_minus_control_snr'] > 5
        assert control['common_exposure_ids'] == ['e0','e1','e2','e3','e4']


def test_stationary_impostor_is_flagged(experiment):
    track_path,judge_path,manifest=experiment
    for p in manifest['trajectories'][0]['positions']:
        p.update(x=30,y=30)
    track_path.write_text(json.dumps(manifest))
    yy,xx=np.mgrid[:61,:61]
    for exposure in manifest['exposures']:
        with fits.open(track_path.parent/exposure['image'],mode='update') as h:
            h[0].data[:]=30+100*gaussian_psf(xx,yy,30,30,1.2)
    result=run_judge(judge_path)
    assert 'a_control_matches_or_exceeds_candidate_on_common_exposures' in result['flags']


def test_single_frame_spike_fails_leave_one_out(experiment):
    track_path,judge_path,manifest=experiment
    for exposure in manifest['exposures'][1:]:
        with fits.open(track_path.parent/exposure['image'],mode='update') as h:
            h[0].data[:]=30
    result=run_judge(judge_path)
    assert 'jackknife_below_requested_snr' in result['flags']
    assert result['constant_flux_gaussian_p'] < 1e-10


def blank_frames(path, manifest):
    for exposure in manifest['exposures']:
        with fits.open(path.parent/exposure['image'],mode='update') as h:
            h[0].data[:]=30


def injection_file(path, **extra):
    value=dict(schema_version=1,photometry_manifest='tracks.json',injection_trajectory_id='mover',reference_fluxes=[0,100],offsets_arcsec=[[0,0],[0,5]])
    value.update(extra)
    result=path.parent/'injections.json'; result.write_text(json.dumps(value)); return result


def test_injections_measure_full_search_recovery_without_editing_inputs(experiment):
    path,_,manifest=experiment; blank_frames(path,manifest)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.glob('*.fits')}
    report=run_injections(injection_file(path))
    assert report['summary'][0]['recovery_fraction'] == 0
    assert report['summary'][1]['recovery_fraction'] == 1
    assert report['summary'][1]['new_recovery_fraction'] == 1
    assert report['summary'][1]['flux_error_in_sigma_percentiles_16_50_84'] == pytest.approx([0,0,0],abs=1e-10)
    assert before == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert all(row['true_trajectory_joint']['trials'] == 2 for row in report['trials'])


def test_preexisting_source_is_not_counted_as_new_recovery(experiment):
    path,_,_=experiment
    report=run_injections(injection_file(path,offsets_arcsec=[[0,0]],reference_fluxes=[0]))
    assert report['summary'][0]['recovery_fraction'] == 1
    assert report['summary'][0]['new_recovery_fraction'] == 0
    assert report['baselines'][0]['detected_before_injection']


def test_offchip_injections_count_as_misses(experiment):
    path,_,_=experiment
    report=run_injections(injection_file(path,offsets_arcsec=[[1000,1000]],reference_fluxes=[100]))
    assert report['summary'][0]['recovery_fraction'] == 0
    assert report['trials'][0]['true_trajectory_joint'] is None


def test_degenerate_orbit_bank_does_not_claim_identification(experiment):
    path,_,manifest=experiment; blank_frames(path,manifest)
    manifest['trajectories'][1]['positions']=manifest['trajectories'][0]['positions']
    path.write_text(json.dumps(manifest))
    result=run_injections(injection_file(path,reference_fluxes=[100]))
    assert result['summary'][0]['recovery_fraction'] == 0
    assert all(row['tied_with_another_trajectory'] for row in result['trials'])


def test_psf_mismatch_affects_injected_flux_recovery(experiment):
    path,_,manifest=experiment; blank_frames(path,manifest)
    result=run_injections(injection_file(path,reference_fluxes=[100],injection_psf_sigma_factor=1.3))
    assert result['trials'][0]['true_trajectory_joint']['flux'] < 90
    assert result['injection_psf_sigma_factor'] == 1.3


def test_sky_offsets_cross_ra_wrap(experiment):
    path,_,_=experiment
    bank=ImageBank(path)
    shifted=bank.shift(bank.tracks['mover'],20,0)
    eid='e0'
    east,north=bank.sky(eid,bank.tracks['mover'][eid]).spherical_offsets_to(bank.sky(eid,shifted[eid]))
    assert east.arcsec == pytest.approx(20,abs=1e-7)
    assert north.arcsec == pytest.approx(0,abs=1e-7)


def test_judge_cli_and_overwrite_protection(experiment):
    _,path,_=experiment; output=path.parent/'report.json'
    args=[sys.executable,'-m','magictools','judge_line',str(path),'-o',str(output)]
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert json.loads(output.read_text())['tool'] == 'judge_line'
    assert subprocess.run(args,capture_output=True).returncode == 2
