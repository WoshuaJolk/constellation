"""Native-response regressions for the expanded archive adapters."""
import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from magictools.frames.adapters import Job, execute, stats
from magictools.frames.credentials import Credentials
from magictools.frames.engine import plan, run
from magictools.frames.extended import boxes
from magictools.frames.geometry import select
from magictools.frames.inventory import coverage
from magictools.frames.legacy import paged_list
from magictools.frames.model import PROVIDERS, mjd, validate
from magictools.frames.profiles import KOA_INSTRUMENTS, NOIRLAB_INSTRUMENTS
from magictools.frames.transport import ArchiveError, Blocked, SameOriginRedirect, Transport

FIX = Path(__file__).parent / "fixtures" / "frames" / "expanded"


def fixture(name):
    b = (FIX/name).read_bytes()
    return (gzip.decompress(b) if name.endswith('.gz') else b).decode()


class Fake:
    def __init__(self, response, credentials=None):
        self.response, self.credentials, self.calls, self.timeout = response, credentials, [], 600

    def request(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response(url, kwargs) if callable(self.response) else self.response


def search(provider, response, **window):
    w = dict(ra=180, dec=0, radius=180, start=mjd('1800-01-01'), end=mjd('2030-01-01'))
    w.update(window)
    job, t, s = Job(provider, 'field', window=w), Fake(response), stats()
    return list(execute(job, {"radius": w['radius']}, t, s)), t, s


@pytest.mark.parametrize('provider,name', [('koa','koa.xml.gz'), ('euclid','euclid.xml.gz'), ('applause','applause.xml.gz'), ('herschel','herschel.xml.gz'), ('jplus','jplus.xml.gz'), ('sdss','sdss.xml.gz')])
def test_native_vo_responses(provider, name):
    frames, t, s = search(provider, fixture(name), ra=180, dec=30, radius=.01 if provider in {'sdss','jplus'} else 180)
    assert frames and s['rejected'] == 0
    assert all(f['id'] and f['url'] for f in frames)
    if provider == 'koa':
        assert parse_qs(urlsplit(t.calls[0]['url']).query)['INSTRUMENT'] == ['DEIMOS']
        assert frames[0]['startMjd'] > 50000
    if provider == 'euclid':
        assert all(f['id'].endswith('.fits') for f in frames)
        assert len({f['key'] for f in frames}) == len(frames)
    if provider == 'sdss':
        assert s['partial'] and all(f['startMjd'] is None for f in frames)
        assert all('fits' in f['url'].lower() for f in frames)
    if provider == 'herschel':
        assert frames[0]['footprint'].startswith('POLYGON ICRS')
        assert frames[0]['endMjd'] > frames[0]['startMjd']


def test_noirlab_families_and_koa_have_separate_resumable_jobs():
    x = validate(dict(mode='field', ra=0, dec=0, radius=.1, start='2000-01-01', end='2026-01-01', providers=['noirlab:mosaic','noirlab:odi','koa']))
    jobs, unavailable = plan(x)
    assert not unavailable
    assert len(jobs) == 5+2+len(KOA_INSTRUMENTS)
    assert len({j.id for j in jobs}) == len(jobs)
    assert {j.params['instrument'] for j in jobs if j.provider == 'koa'} == set(KOA_INSTRUMENTS)


@pytest.mark.parametrize('provider,instrument', [(p,i) for p,names in NOIRLAB_INSTRUMENTS.items() if p!='noirlab' for i in names])
def test_all_noirlab_instrument_aliases(provider, instrument):
    path = FIX/('noirlab-'+instrument+'.json')
    if not path.exists():
        pytest.fail('Missing native-response fixture for '+instrument)
    t, s = Fake(path.read_text()), stats()
    j = Job(provider,'field',dict(ra=0,dec=0,radius=180,start=mjd('1990-01-01'),end=mjd('2030-01-01')),params={'instrument':instrument})
    frames = list(execute(j,{},t,s))
    assert len(frames)+s['rejected']+s['excluded'] == len(json.loads(path.read_text()))-1
    assert all(f['instrument'] == instrument for f in frames)
    assert ['instrument',instrument] in t.calls[0]['data']['search']
    assert not any(term[0]=='proc_type' for term in t.calls[0]['data']['search'])


def test_lco_pagination_and_bad_rows_keep_valid_products():
    data = json.loads(fixture('lco.json'))
    first = {**data,'results':[data['results'][0], {'id':None}], 'next':'https://archive-api.lco.global/frames/?cursor=second'}
    def response(url, kw):
        return json.dumps(data if 'cursor=second' in url else first)
    frames, t, s = search('lco', response, ra=148.8, dec=1, radius=.1, start=mjd('2019-01-01'), end=mjd('2019-02-01'))
    assert len(t.calls) == 2 and len(frames) >= 2 and s['rejected'] == 1
    assert all(f['url'].startswith('https://archive-api.lco.global/frames/') for f in frames)
    assert all(f['footprint'] for f in frames)
    assert parse_qs(urlsplit(t.calls[0]['url']).query)['limit'] == ['100']
    first['next'] = 'https://example.invalid/steal'
    with pytest.raises(ArchiveError, match='pagination origin'):
        search('lco',lambda u,k:json.dumps(first),ra=148.8,dec=1,radius=.1)


def test_gemini_and_akari_malformed_row_isolation():
    rows = json.loads(fixture('gemini.json'))
    rows.append({'filename':None})
    frames, _, s = search('gemini',json.dumps(rows))
    assert len(frames) == len(rows)-1 and s['rejected'] == 1
    frames, _, s = search('akari',fixture('akari.html'))
    assert len(frames) == 4 and s['rejected'] == 1
    assert all(f['productKind']=='observation bundle' for f in frames)


def test_dasch_native_plate_ids_and_point_only_receipt():
    frames, t, s = search('dasch',fixture('dasch.json'),ra=180,dec=30,radius=.5)
    assert frames and s['partial']
    assert all(':exposure:' in f['id'] and ':solution:' in f['id'] for f in frames)
    assert t.calls[0]['data'] == {'ra_deg':180,'dec_deg':30}
    assert frames[0]['retrievalRequest']['method']=='POST'


def test_wfcam_full_result_file_and_hour_angle_query():
    def response(url, kw):
        if 'WSASQL?' in url:
            return '<a id="dl_id" href="http://wsa.roe.ac.uk/tmp/result.xml">Download Results File</a>'
        return fixture('wfcam.xml.gz')
    frames, t, s = search('wfcam',response,ra=181.2,dec=.1,radius=.1,start=mjd('2005-05-18'),end=mjd('2005-05-19'))
    assert frames and s['rejected']==0
    assert len(t.calls)==4  # one query and full result file for each release
    query=parse_qs(urlsplit(t.calls[0]['url']).query)
    assert 'TOP ' not in query['sqlstmt'][0]
    assert 'raBase BETWEEN 12.' in query['sqlstmt'][0]
    assert all(180 < f['ra'] < 183 for f in frames)


def test_splus_pages_past_server_clamped_page_size_and_filters_weights():
    calls=[]
    def response(url, kw):
        p=parse_qs(urlsplit(url).query);skip=int(p['skip'][0]);calls.append(skip)
        return json.dumps([{'id':skip},{'id':skip+1}] if skip<6 else [])
    assert len(list(paged_list(Fake(response),'https://splus.cloud/files')))==6
    assert calls==[0,2,4,6]
    with pytest.raises(ArchiveError,match='repeated a page'):
        list(paged_list(Fake('[{"id":1}]'),'https://splus.cloud/files'))
    rows=json.loads(fixture('splus-files.json'))
    rows.append({**rows[0],'filename':'weight.fits'})
    def files(url, kw):
        p=parse_qs(urlsplit(url).query)
        if int(p['skip'][0]):return '[]'
        return '[{"id":7,"name":"DR4"}]' if '/collections/?' in url else json.dumps(rows)
    frames, _, s=search('splus',files)
    assert len(frames)==3 and s['excluded']==1 and s['partial']
    assert all(f['timeSpanUnknown'] for f in frames)


def test_scanpi_native_scan_identity_utc_and_point_limitation():
    def response(url, kw):
        if 'nph-scanpi?' in url:
            assert 'j2000' in parse_qs(urlsplit(url).query)['locstr'][0]
            return fixture('scanpi-0000.txt.gz')
        band=__import__('re').search(r'/b([1-4])_summary.tbl',url)[1]
        return fixture('scanpi-000'+band+'.txt.gz')
    frames, _, s=search('scanpi',response,ra=180,dec=30,radius=.1,start=mjd('1983-01-01'),end=mjd('1984-01-01'))
    assert frames and s['partial'] and s['rejected']==0
    assert all(f['productKind']=='detector scan' for f in frames)
    assert all('b' in f['id'] and '_' in f['id'] for f in frames)
    assert all(mjd('1983-01-01') <= f['startMjd'] < mjd('1984-01-01') for f in frames)


def test_lasco_native_ids_not_synthetic_image_links():
    frames,t,s=search('lasco',fixture('vso.xml'),ra=75,dec=22,radius=.1,start=mjd('2011-06-07T05:30:00'),end=mjd('2011-06-07T06:00:00'))
    assert len(frames)==4 and s['rejected']==0
    assert all(f['retrievalRequest']['fileid']==f['id'] for f in frames)
    assert all(f['endMjd']>f['startMjd'] for f in frames)
    assert '<instrument>LASCO</instrument>' in t.calls[0]['data']


def test_target_pixels_are_fits_products_not_light_curves():
    def response(url, kw):
        request=json.loads(parse_qs(kw['data'])['request'][0])
        if request['service']=='Mast.Caom.Products':
            return fixture('mast-target-products.json')
        return fixture('mast-target-observations.json')
    frames,t,s=search('mast:targetpixels',response)
    assert frames and all('targ.fits' in f['id'] for f in frames)
    assert all(f['productKind']=='target-pixel time cube' for f in frames)
    assert s['excluded']>0


def test_access_and_service_limits_are_explicit():
    with pytest.raises(Blocked,match='RSP_API_TOKEN'):
        search('rubin','must not call')
    with pytest.raises(Blocked,match='40 arcmin'):
        search('dss','must not call',radius=1)
    frames, _, s=search('dss',fixture('dss.xml.gz'),radius=.05)
    assert frames and s['partial'] and all(f['productKind']=='cutout' for f in frames)


def test_wrap_and_polar_boxes_cover_cones():
    b=list(boxes(359.9,0,.2))
    assert any(left==0 for left,_,_,_ in b)
    assert any(right==360 for _,right,_,_ in b)
    b=list(boxes(0,89,2))
    assert sum(right-left for left,right,_,_ in b)==360


def test_combined_products_do_not_make_false_ephemeris_exclusions():
    w={'ra':0,'dec':0,'radius':1,'start':50000,'end':50001,'points':[{'mjd':50000,'ra':0,'dec':0,'sigmaArcsec':0},{'mjd':50001,'ra':1,'dec':0,'sigmaArcsec':0}]}
    f={'startMjd':40000,'timeSpanUnknown':True,'productKind':'stack'}
    assert select(f,w,{})
    assert 'combined product' in f['match']


def test_inventory_maps_every_entry_without_claiming_full_coverage():
    report=coverage([{'provider':'euclid','status':'ok'}])
    assert report['datasetCount']==51 and report['providerCount']==39
    assert not report['allAstronomicalObservationsCovered']
    assert all(set(r['providers'])<=set(PROVIDERS) for r in report['datasets'])
    assert next(r for r in report['datasets'] if r['dataset']=='INT WFC')['providerQueryStates']['iphas']==['not queried']
    assert next(r for r in report['datasets'] if r['dataset']=='Euclid Q1')['providerQueryStates']['euclid']==['ok']


def test_cross_origin_redirect_strips_all_archive_secrets():
    from urllib.request import Request
    req=Request('https://archive.gemini.edu/data',headers={'Authorization':'secret','Cookie':'session=secret','X-api-key':'secret'})
    new=SameOriginRedirect().redirect_request(req,None,302,'redirect',{},'https://example.invalid/file')
    assert not new.has_header('Authorization') and not new.has_header('Cookie') and not new.has_header('X-api-key')


def test_transport_gzip_and_login_response_never_persist_secrets(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    class Response(io.BytesIO):
        status=200
        headers={"Content-Type":"application/json"}
    payloads=[gzip.compress(b'{"ok":true}'),b'{"access_token":"newly-issued-secret"}']
    def opener():
        return SimpleNamespace(open=lambda *a,**k:Response(payloads.pop(0)))
    monkeypatch.setattr('urllib.request.build_opener',lambda *a:opener())
    t=Transport(tmp_path)
    assert t.request('https://example.invalid/data')=='{"ok":true}'
    assert t.calls[0]['responseEncoding']=='gzip'
    result=t.request('https://splus.cloud/adss/v1/auth/login',private_data='password=hidden',sensitive_response=True)
    assert json.loads(result)['access_token']=='newly-issued-secret'
    assert t.calls[1]['responseRedacted'] and 'responseFile' not in t.calls[1]
    assert 'password' not in json.dumps(t.calls)
    assert all(b'newly-issued-secret' not in p.read_bytes() and b'hidden' not in p.read_bytes() for p in tmp_path.iterdir())


def test_missing_dates_are_partial_without_endless_date_subdivision(tmp_path):
    class Factory(Fake):
        def __init__(self,directory,timeout,credentials):
            super().__init__(fixture('sdss.xml.gz'),credentials)
    x=validate(dict(mode='field',ra=180,dec=30,radius=.01,start='2000-01-01',end='2026-01-01',providers=['sdss']))
    result=run(x,tmp_path,transport_factory=Factory)
    assert result['status']=='incomplete'
    assert len(result['checks'])==1 and result['checks'][0]['status']=='partial'
    assert result['frames']


def test_igaps_unavailable_is_not_zero_coverage(tmp_path):
    class Factory(Fake):
        def __init__(self,directory,timeout,credentials):
            super().__init__('',credentials)
        def request(self,url,**kwargs):
            self.calls.append({'url':url})
            raise ArchiveError('Archive returned HTTP 503')
    x=validate(dict(mode='field',ra=180,dec=30,radius=.1,start='2000-01-01',end='2026-01-01',providers=['iphas']))
    result=run(x,tmp_path,transport_factory=Factory)
    assert result['status']=='incomplete' and result['checks'][0]['status']=='error'
    assert not result['frames'] and result['checks'][0]['queryUrl'].endswith('igaps-images.fits.gz')


def test_igaps_real_fits_metadata_has_utc_epochs():
    def response(url,kw):
        assert kw['binary'] is True
        return FIX/'igaps.fits'
    frames,t,s=search('iphas',response)
    assert len(frames)==3 and s['received']==3 and s['rejected']==0
    assert all(mjd('2003-08-08') < f['startMjd'] < mjd('2003-08-09') for f in frames)
    assert all(f['url'].endswith('.fits.fz') for f in frames)


def test_metadata_cache_retains_artifacts_without_requery(tmp_path,monkeypatch):
    import io
    from types import SimpleNamespace
    class Response(io.BytesIO):
        status=200
        headers={'Content-Type':'text/plain'}
    calls=[]
    def fetch(*a,**k):
        calls.append(a[0].full_url)
        return Response(b'public metadata')
    monkeypatch.setattr('urllib.request.build_opener',lambda *a:SimpleNamespace(open=fetch))
    for name in ['first','second']:
        t=Transport(tmp_path/name)
        t.shared_cache=tmp_path/'cache'
        assert t.cached_request('http://example.invalid/index')=='public metadata'
        assert (t.directory/t.calls[-1]['responseFile']).read_bytes()==b'public metadata'
    assert len(calls)==1 and t.calls[-1]['cacheHit']


def test_fractional_archive_timestamps_keep_utc_epoch():
    expected=mjd('2003-08-08')+(22*3600+7*60+10.4)/86400
    for fraction in ['4','40','400','4000','40000','400000','400000000']:
        assert mjd('2003-08-08T22:07:10.'+fraction+'Z') == pytest.approx(expected,abs=1e-10,rel=0)
        assert mjd('2003-08-08T23:07:10.'+fraction+'+01:00') == pytest.approx(expected,abs=1e-10,rel=0)
