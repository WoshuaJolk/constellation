"""Native protocol fixtures and regressions for evidence, identity, and access."""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from magictools.frames.adapters import Job, execute, stats
from magictools.frames.assessment import audit_run
from magictools.frames.credentials import Credentials
from magictools.frames.engine import plan, run, merge_manifests
from magictools.frames.geometry import region_intersects, select
from magictools.frames.model import PROVIDERS, mjd, validate
from magictools.frames.parsers import votable
from magictools.frames.profiles import VO_PROFILES
from magictools.frames.transport import Blocked
from magictools.frames.vo import datalinks, discover, endpoint_url, inspect_service, service_auth

FIX = Path(__file__).parent / 'fixtures' / 'frames' / 'vo'


def fixture(name):
    return gzip.decompress((FIX/(name+'.response.gz')).read_bytes()).decode()


class Fake:
    def __init__(self, directory=None, timeout=600, credentials=None, response=''):
        self.directory, self.timeout, self.credentials = directory, timeout, credentials
        self.calls, self.response = [], response

    def request(self, url, **kwargs):
        self.calls.append({'url': url, **kwargs})
        return self.response(url, kwargs) if callable(self.response) else self.response


@pytest.mark.parametrize('provider', list(VO_PROFILES)+['kmtnet', 'miris'])
def test_native_new_archives_retain_scientific_metadata(provider):
    s = stats()
    t = Fake(response=fixture(provider.replace(':', '-')))
    w = dict(ra=180, dec=0, radius=180, start=mjd('1800-01-01'), end=mjd('2030-01-01'))
    # Avoid all-sky SIA1 tiling in fixture tests; this native field is the Crab.
    if VO_PROFILES.get(provider, {}).get('protocol') == 'sia1':
        w.update(ra=83.633, dec=22.0145, radius=.01)
    frames = list(execute(Job(provider, 'field', window=w), {'radius': w['radius']}, t, s))
    assert frames and s['rejected'] == 0
    assert all(f['id'] and f['url'] for f in frames)
    assert s['received'] == len(frames) + s['excluded']
    if provider == 'alma':
        assert len({f['key'] for f in frames}) == len(frames)
        assert all(f['id'].startswith('uid://') and f['linkKind'] == 'DataLink' for f in frames)
        assert all(f['timeSpanUnknown'] for f in frames)
    if provider == 'casda':
        assert len({f['key'] for f in frames}) == len(frames)
    if provider == 'heasarc':
        assert frames[0]['footprint'].strip().startswith('CIRCLE')
        assert region_intersects(frames[0]['footprint'], 0, 0, 0) is not None
    if provider == 'chandra':
        assert frames[0]['exposureSeconds'] == pytest.approx(float(frames[0]['raw']['exptime'])*1000)
        assert frames[0]['fovRadius'] > .05
    if provider.startswith('xmm'):
        assert all(urlsplit(f['url']).hostname == 'nxsa.esac.esa.int' for f in frames)
        assert all(f['endMjd'] > f['startMjd'] for f in frames)
    if provider == 'nrao':
        assert all(f['endMjd'] is None and f['timeSpanUnknown'] for f in frames)
        assert all(f['retrieval']['requiresArchiveSelection'] for f in frames)
    if provider in {'kmtnet', 'miris'}:
        assert s['partial'] and not s['canSplitTime']
        assert all(f['exposureSeconds'] > 0 and f['fovRadius'] > .5 for f in frames)


def manifest(**updates):
    value = dict(mode='field', ra=10, dec=20, radius=.1, start='2020-01-01', end='2020-01-03',
                 providers=['vo:test'], services={'vo:test': {'protocol': 'obscore', 'endpoint': 'https://example.invalid/tap/sync'}})
    value.update(updates)
    return validate(value)


def table(rows='', status='OK', extra=''):
    names = ['dataproduct_type', 'obs_id', 'obs_publisher_did', 'access_url', 's_ra', 's_dec', 't_min', 't_max', 's_region']
    columns = ''.join(f'<FIELD name="{n}" datatype="char" arraysize="*"/>' for n in names)
    marker = f'<INFO name="QUERY_STATUS" value="{status}"/>' if status else ''
    return f'<VOTABLE><RESOURCE>{marker}<TABLE>{columns}<DATA><TABLEDATA>{rows}</TABLEDATA></DATA></TABLE>{extra}</RESOURCE></VOTABLE>'


def row(did='product-a', start='58849', stop='58849.01', footprint='CIRCLE ICRS 10 20 .2', obs_id='shared-visit'):
    return '<TR>'+''.join(f'<TD>{v}</TD>' for v in ['image', obs_id, did, 'https://example.invalid/'+did+'.fits', 10, 20, start, stop, footprint])+'</TR>'


def factory(response):
    class Factory(Fake):
        def __init__(self, directory, timeout, credentials):
            super().__init__(directory, timeout, credentials, response)
    return Factory


def test_custom_services_do_not_mutate_registry_and_change_job_identity():
    before = set(PROVIDERS)
    x = manifest()
    jobs, unqueried = plan(x)
    assert not unqueried and jobs[0].params['service']['protocol'] == 'obscore'
    x['services']['vo:test']['endpoint'] = 'https://second.invalid/tap/sync'
    assert plan(x)[0][0].id != jobs[0].id
    assert set(PROVIDERS) == before and 'vo:test' not in PROVIDERS
    assert 'vo:test' in manifest(providers='all')['providers']


@pytest.mark.parametrize('config', [
    {'protocol':'sql'}, {'endpoint':'file:///etc/passwd'}, {'endpoint':'https://u:p@example.invalid/tap'},
    {'endpoint':'https://example.invalid/tap?token=secret'}, {'table':'a; DROP TABLE b'},
    {'productTypes':['timeseries']}, {'collections':'ALMA'}, {'maxrec':0}, {'headers':{'Authorization':'secret'}},
])
def test_invalid_service_options_rejected_before_io(config):
    with pytest.raises(ValueError):
        manifest(services={'vo:test':{'protocol':'obscore','endpoint':'https://example.invalid/tap/sync',**config}})


def test_endpoint_parameters_preserve_collections_and_quote_names():
    u = endpoint_url('https://example.invalid/sia?COLLECTION=my%20survey&POS=old&', {'POS':'CIRCLE 1 2 3'})
    assert parse_qs(urlsplit(u).query) == {'COLLECTION':['my survey'],'POS':['CIRCLE 1 2 3']}
    from magictools.frames.vo import obscore_query
    x=manifest(); w=plan(x)[0][0].window
    query=obscore_query({'collections':["O'Brien"], 'productTypes':['image','cube']},w)
    assert "'O''Brien'" in query and 't_min IS NULL' in query and 't_max<t_min' in query
    assert 's_region IS NULL' in query and 's_fov/2' in query


def test_distinct_products_with_shared_visit_and_missing_times_survive(tmp_path):
    result=run(manifest(),tmp_path,transport_factory=factory(table(row()+row('product-b',start='',stop=''))))
    assert len(result['frames']) == 2 and result['status'] == 'complete'
    a=result['checks'][0]['coverageAssessment']
    assert a['querySucceeded'] and a['indexExhausted']
    assert a['returnedProductEvidence']['utcIntervals'] == 1
    assert a['outcome'] == 'evidence-found' and a['entireRequestedRegionCovered'] is None
    assert a['gaps'] and not a['allObservationsCovered']
    assert merge_manifests([result])['coverageAssessment']['providers']['vo:test']['queries'] == 1


def test_existing_obscore_keeps_distinct_visit_products_and_native_frame_dedupe():
    from magictools.frames.adapters import obscore
    j=Job('cadc','field')
    shared={'obs_id':'visit-123','instrument_name':'camera','t_min':'58849'}
    a=obscore(j,{**shared,'obs_publisher_did':'ivo://archive/product-a'})
    b=obscore(j,{**shared,'obs_publisher_did':'ivo://archive/product-b'})
    assert a['key'] != b['key']
    native=obscore(j,{**shared,'obs_id':'1234567o','instrument_name':'MegaCam','obs_publisher_did':'ivo://archive/1234567o'})
    assert native['key']=='cfht/megacam:1234567'
    stack=obscore(j,{**shared,'calib_level':'3'})
    assert stack['timeSpanUnknown']


def test_empty_http_success_requires_a_valid_schema_and_status(tmp_path):
    r=run(manifest(),tmp_path/'ok',transport_factory=factory(table()))
    assert r['checks'][0]['coverageAssessment']['outcome'] == 'no-matches-in-queried-index'
    r=run(manifest(),tmp_path/'unknown',transport_factory=factory(table(status='')))
    assert r['status']=='incomplete'
    assert r['checks'][0]['coverageAssessment']['outcome']=='unverified'
    r=run(manifest(),tmp_path/'invalid',transport_factory=factory('<VOTABLE><RESOURCE><TABLE><FIELD name="login"/></TABLE></RESOURCE></VOTABLE>'))
    assert r['checks'][0]['status']=='error'


def test_malformed_rows_are_counted_without_losing_good_rows(tmp_path):
    malformed='<TR><TD>too few</TD></TR>'
    r=run(manifest(),tmp_path,transport_factory=factory(table(row()+malformed)))
    assert len(r['frames'])==1 and r['checks'][0]['rejectedRows']==1
    assert r['checks'][0]['status']=='partial'


def test_overflow_subdivides_obscore_but_not_sia1(tmp_path):
    def response(url,kwargs):
        query=parse_qs(urlsplit(url).query)['QUERY'][0]
        return table(status='OVERFLOW' if 't_max>=58849.0' in query and 't_min<58851.0' in query else 'OK')
    r=run(manifest(),tmp_path/'tap',transport_factory=factory(response))
    assert [c['status'] for c in r['checks']]==['split','ok','ok']
    x=manifest(services={'vo:test':{'protocol':'sia1','endpoint':'https://example.invalid/sia'}})
    r=run(x,tmp_path/'sia',transport_factory=factory(table(status='OVERFLOW')))
    assert len(r['checks'])==1 and r['checks'][0]['status']=='limited'


def test_sia2_uses_space_separated_time_and_filters_products(tmp_path):
    x=manifest(services={'vo:test':{'protocol':'sia2','endpoint':'https://example.invalid/sia?COLLECTION=public'}})
    r=run(x,tmp_path,transport_factory=factory(table(row())))
    p=parse_qs(urlsplit(r['checks'][0]['requests'][0]['url']).query)
    assert p['TIME']==['58849.0 58851.0'] and p['DPTYPE']==['image','cube']
    assert p['COLLECTION']==['public']


def test_custom_credentials_require_private_origin_grant(tmp_path,monkeypatch):
    monkeypatch.setenv('VO_TEST_TOKEN','never-in-receipt')
    path=tmp_path/'credentials.json'
    path.write_text(json.dumps({'providers':{'custom':{'token':{'env':'VO_TEST_TOKEN'},'allowedOrigins':['https://example.invalid']}}}))
    path.chmod(0o600)
    c=Credentials(path); t=Fake(credentials=c)
    config={'protocol':'obscore','endpoint':'https://example.invalid/tap/sync','credentialProfile':'custom'}
    assert service_auth(config,t)=={'token':'never-in-receipt'}
    assert c.status()['custom']['configured']
    assert 'never-in-receipt' not in json.dumps(c.status())
    for endpoint in ['https://other.invalid/tap/sync','http://example.invalid/tap/sync','https://example.invalid:444/tap/sync']:
        with pytest.raises(Blocked):service_auth({**config,'endpoint':endpoint},t)
    assert service_auth({'endpoint':'https://example.invalid'},t)=={}


def test_registry_and_schema_inspection_use_native_responses():
    t=Fake(response=fixture('registry'))
    result=discover(t,'Chandra')
    assert result['candidates'] and all(not c['verified'] for c in result['candidates'])
    assert any(c['service']['protocol']=='sia2' for c in result['candidates'])
    result=inspect_service(Fake(response=fixture('inspect')),VO_PROFILES['alma'])
    assert result['schemaCompatible'] and not result['coverageEstablished']


def test_datalink_keeps_access_methods_and_errors():
    result=datalinks(Fake(response=fixture('links')),'https://almascience.org/datalink/sync?ID=test')
    assert len(result['links'])==7
    assert all('semantics' in r and 'service_def' in r for r in result['links'])
    assert any(r['access_url'] for r in result['links'])


def test_binary_votable_array_masks_do_not_abort_provider():
    s=stats();rows=list(votable(fixture('binary-arrays'),s))
    assert len(rows)==3 and s['rejected']==0


def test_trail_intersection_survives_midpoint_outside_and_temporal_edges():
    points=[{'mjd':60000,'ra':0,'dec':0,'sigmaArcsec':0},{'mjd':60001,'ra':10,'dec':0,'sigmaArcsec':0}]
    w={'start':60000,'end':60001,'points':points,'label':'fast'}
    x={'radius':0,'sigmaMultiplier':1,'eellipse':0,'observer':'geocenter'}
    f={'startMjd':60000,'exposureSeconds':3600,'timeScale':'UTC','footprint':'CIRCLE ICRS .02 0 .01'}
    assert select(f,w,x) and 'exposure-track' in f['match']
    assert f['predictions'][0]['exposureEnvelopeRadiusDegrees']>.2
    assert not select({**f,'footprint':'CIRCLE ICRS 90 0 .01'},w,x)
    assert select({**f,'startMjd':59999.99},w,x)
    assert region_intersects('CIRCLE ICRS 10 100 .1',10,20,.1) is None
    assert region_intersects({'malformed':'geometry'},10,20,.1) is None


def test_artifact_checksum_audit_detects_tampering(tmp_path):
    class Recorded(Fake):
        def request(self,url,**kwargs):
            payload=table(row()).encode();path=Path(self.directory)/'raw.xml';path.write_bytes(payload)
            self.calls.append({'url':url,'responseFile':'raw.xml','sha256':hashlib.sha256(payload).hexdigest()})
            return payload.decode()
    run(manifest(),tmp_path,transport_factory=Recorded)
    a=audit_run(tmp_path,True)
    assert a['artifactIntegrity']['verified'] and a['artifactIntegrity']['checked']==1
    next((tmp_path/'queries').rglob('raw.xml')).write_text('tampered')
    a=audit_run(tmp_path,True)
    assert not a['artifactIntegrity']['verified'] and len(a['artifactIntegrity']['failures'])==1
