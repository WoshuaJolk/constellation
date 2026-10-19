import json

from astropy.io import fits
import numpy as np
import pytest

from magictools.photometry import sampled_psf
from magictools.stamps import run_stamp, weigh_stamp


def test_measured_psf_model_favors_trailed_source():
    ky,kx=np.mgrid[-5:6,-5:6]
    kernel=np.exp(-(kx**2/2+ky**2/4))+.3*np.exp(-((kx-2)**2+(ky+1)**2))
    yy,xx=np.mgrid[:31,:31]
    data=2000*sampled_psf(xx,yy,15.2,15.1,kernel,7,2)+20
    report=weigh_stamp(data,4,15.2,15.1,1.2,7,2,3000,psf_kernel=kernel)
    assert report['preferred_model'] == 'moving_source'
    assert report['point_source_psf'] == 'native_pixel_empirical'
    assert report['moving_source_profile_chi2'] == pytest.approx(0,abs=1e-8)
    no_trail=weigh_stamp(data,4,15.2,15.1,1.2,0,0,3000,psf_kernel=kernel)
    assert no_trail['models']['stationary_point']['log_bayes_factor_moving_vs_model'] == 0


def test_independent_artifact_template_and_uniform_bank_evidence():
    artifact=np.zeros((25,25)); artifact[7:18,9]=1; artifact[17,9:18]=1
    data=1000*artifact/artifact.sum()+5
    result=weigh_stamp(data,1,12,12,1.3,5,0,1000,empirical_artifacts={'corner':artifact})
    assert result['preferred_model'] == 'empirical_artifact'
    twice=weigh_stamp(data,1,12,12,1.3,5,0,1000,empirical_artifacts={'one':artifact,'two':artifact*5})
    assert twice['models']['empirical_artifact']['log_evidence_vs_background'] == pytest.approx(result['models']['empirical_artifact']['log_evidence_vs_background'])


def test_full_frame_cutout_and_psf_provenance(tmp_path):
    yy,xx=np.mgrid[:100,:100]; ky,kx=np.mgrid[-5:6,-5:6]
    kernel=np.exp(-(kx**2+ky**2)/2)
    data=300*sampled_psf(xx,yy,50.2,40.1,kernel,5,1)+10
    fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(data,name='SCI'),fits.ImageHDU(np.full_like(data,4),name='VAR'),fits.ImageHDU(kernel,name='PSF')]).writeto(tmp_path/'image.fits')
    value=dict(schema_version=1,image='image.fits',image_hdu='SCI',variance='image.fits',variance_hdu='VAR',psf_image='image.fits',psf_hdu='PSF',x=50.2,y=40.1,psf_sigma_pixels=1,trail_dx=5,trail_dy=1,flux_prior_sigma=500,stamp_radius_pixels=15)
    path=tmp_path/'input.json'; path.write_text(json.dumps(value))
    result=run_stamp(path)
    assert result['stamp_bounds'] == dict(x_start=35,x_stop=67,y_start=25,y_stop=57)
    assert result['preferred_model'] == 'moving_source'
    assert len(result['provenance']['inputs']) == 2


def test_template_must_match_fitted_stamp():
    with pytest.raises(ValueError,match='matching the stamp'):
        weigh_stamp(np.ones((21,21)),1,10,10,1,0,0,100,empirical_artifacts={'bad':np.ones((20,20))})
