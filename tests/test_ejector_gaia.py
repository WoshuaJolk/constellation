import numpy as np
import pytest
from astropy.table import Table
from magictools.ejector_gaia import REQUIRED, distance_draws, gaia_joint_draws, read_gaia

FRAME = dict(galcen_distance_pc=8122., z_sun_pc=20.8, galcen_v_sun_kms=[12.9,245.6,7.78])
PRIORS = dict(distance_length_pc=1350., missing_rv_mean_kms=0., missing_rv_sigma_kms=300., parallax_zero_point_mas=-.017, shared_zero_point_sigma_mas=.01)


def catalogue():
    rows = []
    for i in range(3):
        row = {key: 0. for key in REQUIRED}
        row.update(source_id=str(100+i), ref_epoch=2016., ra=100., dec=30., parallax=10., pmra=10., pmdec=-5.)
        for key in REQUIRED:
            if key.endswith('_error'): row[key] = .01
        row.update(radial_velocity=np.nan, radial_velocity_error=np.nan, rv_nb_transits=0)
        rows.append(row)
    return Table(rows=rows)


def test_distance_high_snr_and_negative_parallax_have_positive_distances():
    d = distance_draws(10., .001, np.zeros(3), 1350, np.array([.16,.5,.84]))
    assert d[1] == pytest.approx(100, rel=1e-5)
    assert d[2] - d[0] == pytest.approx(.02, rel=.02)
    d = distance_draws(-1., .5, np.zeros(3), 1350, np.array([.16,.5,.84]))
    assert np.all(np.diff(d) > 0) and d[0] > 0


def test_missing_errors_reject_rows_instead_of_treating_exact():
    table = catalogue()
    table.remove_column('pmra_error')
    states, audit = gaia_joint_draws(table, 16, 8, FRAME, PRIORS)
    assert states.shape == (16, 0, 6)
    assert audit['excluded_rows'] == 3
    assert all('pmra_error' in r['reason'] for r in audit['ledger'])


def test_rv_prior_not_zero_fill_and_usable_measurement_preserved():
    table = catalogue()
    table['radial_velocity'][1] = 20.
    table['radial_velocity_error'][1] = 1.
    table['rv_nb_transits'][1] = 10
    states, audit = gaia_joint_draws(table, 1024, 8, FRAME, PRIORS)
    assert audit['ledger'][0]['radial_velocity'] == 'explicit_prior'
    assert audit['ledger'][1]['radial_velocity'] == 'measurement'
    assert np.linalg.norm(states[:, 0, 3:].std(axis=0)) > 250
    assert np.linalg.norm(states[:, 1, 3:].std(axis=0)) < 2
    assert audit['included_rows'] == 3


def test_correlations_and_global_calibration_survive_sampling():
    table = catalogue()
    table['parallax_pmra_corr'] = .9
    states, audit = gaia_joint_draws(table, 1024, 42, FRAME, PRIORS)
    # Shared zero point induces common distance offsets despite independent noise.
    assert np.corrcoef(states[:, 0, 0], states[:, 1, 0])[0,1] > .4
    assert len(audit['shared_zero_point_draws_mas']) == 1024
    table['ra_dec_corr'][0] = 1.2
    _, audit = gaia_joint_draws(table, 16, 42, FRAME, PRIORS)
    assert not audit['ledger'][0]['included']


def test_overflow_response_is_rejected(tmp_path):
    path=tmp_path/'truncated.vot'
    path.write_text('<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="OK"/><INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE></VOTABLE>')
    with pytest.raises(ValueError, match='truncated'):
        read_gaia(path)


def test_real_web_field_votable_names_and_exact_input_ids():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'research/ejector_catalogue/data/web_field_enriched.vot'
    table=read_gaia(path)
    assert len(table)==162
    assert 'source_id' in table.colnames
    states,audit=gaia_joint_draws(table,16,20261019,FRAME,PRIORS)
    assert states.shape==(16,158,6)
    assert audit['excluded_rows']==4
    assert all(not row['included'] or row['radial_velocity'] in {'measurement','explicit_prior'} for row in audit['ledger'])


def test_archived_allsky_query_is_complete_and_checksum_matches():
    import hashlib,json
    from pathlib import Path
    directory=Path(__file__).resolve().parents[1]/'research/ejector_catalogue/data'
    manifest=json.loads((directory/'download_manifest.json').read_text())
    source=directory/'gaia_fast.fits.gz'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['merged_sha256']
    assert [r['partition'] for r in manifest['partitions']]==list(range(192))
    assert all(r['query_status']=='OK' for r in manifest['partitions'])
    assert sum(r['rows'] for r in manifest['partitions'])==38889
    table=read_gaia(source)
    assert len(table)==len(set(table['source_id']))==38889
    for left,right in zip(manifest['partitions'][:-1],manifest['partitions'][1:]):
        assert left['high_exclusive']==right['low_inclusive']
