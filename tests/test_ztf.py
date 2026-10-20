import io
import json

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
import numpy as np
import pytest

from magictools import ztf
from magictools.io import run_photometry


@pytest.fixture
def products():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [21, 21]; wcs.wcs.crval = [180, 30]
    wcs.wcs.cdelt = [-1/3600, 1/3600]; wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    header = wcs.to_header()
    header.update(DBPID=757381141115, FILTERID=2, OBSJD=2458511.8811458, GAIN=6.2, MAGZP=26.3,
                  SEEING=2.4, EXPTIME=30., SATURATE=50000., OBSLON=-116.8598, OBSLAT=33.3573, OBSALT=1668.,
                  SHUTOPEN='2019-01-28T09:08:52.017', SHUTCLSD='2019-01-28T09:09:22.005')
    rng = np.random.default_rng(42)
    science = rng.normal(400, 12, (41, 41)); mask = np.zeros((41, 41), dtype=np.int16); mask[10, 10] = 2
    yy, xx = np.mgrid[-5:6, -5:6]; kernel = np.exp(-(xx**2 + yy**2)/2)
    def body(data, hdr=None):
        stream = io.BytesIO(); fits.writeto(stream, data, hdr); return stream.getvalue()
    row = dict(pid='757381141115', obsjd=2458511.8811458, exptime=30., filefracday=20190128381134,
               field=671, ccdid=3, qid=4, filtercode='zr', infobits=0, maglimit=20.343)
    return [body(science, header), body(mask, header), body(kernel)], row


def table_bytes(rows):
    table = Table(rows=rows)
    table.meta['keywords'] = {'RowsRetrieved': {'value': len(rows)}, 'QUERY_STATUS': {'value': 'OK'}}
    stream = io.StringIO(); table.write(stream, format='ascii.ipac'); return stream.getvalue().encode()


def test_index_parsing_and_path(products):
    _, row = products
    result = ztf.parse_index(table_bytes([row]))
    assert result[0]['pid'] == row['pid']
    assert ztf.product_url(row, 'sciimg.fits').endswith('/2019/0128/381134/ztf_20190128381134_000671_zr_c03_o_q4_sciimg.fits')


def test_search_deduplicates_records_query_and_fails_instead_of_truncating(products, tmp_path, monkeypatch):
    _, row = products
    monkeypatch.setattr(ztf, '_download', lambda url, limit: table_bytes([row]))
    value = dict(schema_version=1, regions=[dict(ra_deg=180,dec_deg=30,width_deg=.1)]*2,
                 start_mjd_utc=58511, end_mjd_utc=58512, filtercode='zr', max_exposures=1)
    path = tmp_path/'query.json'; path.write_text(json.dumps(value))
    result = ztf.run_ztf_search(path)
    assert len(result['exposures']) == 1 and len(result['requests']) == 2
    assert 'INTERSECT=OVERLAPS' in result['requests'][0]['url']
    assert len(result['requests'][0]['response_sha256']) == 64
    monkeypatch.setattr(ztf, '_download', lambda url, limit: table_bytes([row, {**row, 'pid': '757381141116'}]))
    with pytest.raises(ValueError, match='nothing was truncated'):
        ztf.run_ztf_search(path)


def test_reject_overflow_and_inconsistent_count(products):
    _, row = products
    body = table_bytes([row])
    with pytest.raises(ValueError, match='QUERY_STATUS'):
        ztf.parse_index(body.replace(b'OK', b'OVERFLOW'))
    with pytest.raises(ValueError, match='row count'):
        ztf.parse_index(body.replace(b'RowsRetrieved=1', b'RowsRetrieved=2'))


def test_calibration_uses_shutter_midpoint_native_seeing_and_mask(products):
    bodies, row = products
    exp, notes, hdus = ztf.prepare_exposure(*bodies, row, 25)
    expected = Time('2019-01-28T09:09:07.011', scale='utc').tdb.mjd
    assert exp['mjd_tdb'] == pytest.approx(expected, abs=1e-10)
    assert exp['exposure_seconds'] == pytest.approx(29.988, abs=1e-7)
    assert exp['psf_sigma_pixels'] == pytest.approx(2.4/2.354820045)
    assert exp['flux_scale'] == pytest.approx(10**(.4*1.3))
    assert hdus['MASK'].data[10,10] == 1
    assert notes['sky_sigma_adu'] == pytest.approx(12, rel=.08)
    assert np.all(hdus['VAR'].data > 0)


def test_wrong_product_and_misaligned_mask_rejected(products):
    bodies, row = products
    with pytest.raises(ValueError, match='product ID'):
        ztf.prepare_exposure(*bodies, {**row, 'pid': 'wrong'}, 25)
    with fits.open(io.BytesIO(bodies[1])) as hdus:
        hdus[0].header['CRPIX1'] += 1
        stream = io.BytesIO(); hdus.writeto(stream)
    with pytest.raises(ValueError, match='WCS do not align'):
        ztf.prepare_exposure(bodies[0], stream.getvalue(), bodies[2], row, 25)


def test_fetch_to_portable_force_track_manifest(products, tmp_path, monkeypatch):
    bodies, row = products
    (tmp_path/'index.json').write_text(json.dumps(dict(schema_version=1,tool='ztf_search',exposures=[row])))
    def download(url, limit):
        return bodies[2 if 'psfcent' in url else 1 if 'mskimg' in url else 0]
    monkeypatch.setattr(ztf, '_download', download)
    value = dict(schema_version=1,index='index.json',directory='data',reference_mag=25,
                 selections=[dict(pid=row['pid'],ra_deg=180,dec_deg=30,size_arcsec=60)])
    path=tmp_path/'fetch.json'; path.write_text(json.dumps(value))
    result = ztf.run_ztf_fetch(path)
    assert len(result['products']) == 3
    manifest=dict(schema_version=1,exposures=result['photometry_exposures'],trajectories=[dict(id='center',positions=[dict(exposure_id=result['exposures'][0]['id'],x=20,y=20)])])
    track_path=tmp_path/'track.json'; track_path.write_text(json.dumps(manifest))
    assert run_photometry(track_path)['trajectories'][0]['measurements'][0]['psf_model'] == 'native_pixel_empirical'
    with pytest.raises(FileExistsError):
        ztf.run_ztf_fetch(path)


def test_site_parallax_has_earth_radius_and_rotates():
    from magictools.orbits import observatory_positions
    class Earth:
        def earth_position(self, times):
            return np.zeros((len(times), 3))
    result = observatory_positions(Earth(), [60000, 60000.25], dict(longitude_deg=0,latitude_deg=0,height_m=0))
    np.testing.assert_allclose(np.linalg.norm(result,axis=1)*149597870700, 6378137, rtol=1e-10)
    assert abs(np.dot(result[0],result[1]) / np.linalg.norm(result[0])**2) < .01
