import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from magictools.frames.adapters import Job, execute, ssois_params, most_params, stats
from magictools.frames.credentials import Credentials, initialize
from magictools.frames.engine import run, plan, merge_manifests, status
from magictools.frames.geometry import distance, interpolate, polygon_intersects, select
from magictools.frames.model import validate, read_ephemeris, mjd
from magictools.frames.parsers import canonical
from magictools.frames.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "frames"
FIELD = {"mode": "field", "ra": 180, "dec": 30, "radius": .05, "start": "2019-01-28", "end": "2019-01-29", "providers": ["ztf"]}
ORBIT = {"label": "nominal", "epochMjd": 55200, "a": 2.7413806, "e": .0855139, "i": 6.98707, "node": 43.05408, "peri": 120.67443, "meanAnomaly": 206.88213}


def transport(body, error=None):
    class Fake:
        def __init__(self, directory, timeout, credentials):
            self.calls = []
            self.credentials = credentials
            self.timeout = timeout

        def request(self, url, **kwargs):
            self.calls.append({"url": url, "body": kwargs.get("data"), "timeoutSeconds": self.timeout})
            if error:
                raise error
            return body(url, kwargs) if callable(body) else body
    return Fake


def test_census_limits_and_mpc_resolver():
    x = validate({"mode": "orbits", "start": "2000-01-01", "end": "2026-12-31", "timeoutSeconds": 600,
                  "eellipse": .51*3600, "orbits": [{**ORBIT, "label": str(i)} for i in range(64)], "providers": ["ssois"]})
    jobs, unavailable = plan(x)
    assert len(jobs) == 64 and not unavailable
    p = ssois_params(jobs[0], x)
    assert p["eellipse"] == .51*3600 and p["epoch1"] == "2000-01-01"
    name = validate({**FIELD, "mode": "object", "object": "2016 FV238", "resolver": "bynameMPC", "providers": ["ssois"]})
    assert ssois_params(plan(name)[0][0], name)["search"] == "bynameMPC"


def test_s60_complete_response_is_not_truncated(tmp_path):
    text = gzip.decompress((FIXTURES / "s60.tsv.gz").read_bytes()).decode()
    x = validate({"mode": "orbits", "start": "2000-01-01", "end": "2026-12-31", "providers": ["ssois"], "orbits": [ORBIT]})
    result = run(x, tmp_path / "run", transport_factory=transport(text))
    assert result["checks"][0]["receivedRows"] == 12218
    assert result["checks"][0]["count"] == 12218
    assert result["checks"][0]["rejectedRows"] == 0
    assert len(result["frames"]) > 12000
    assert result["status"] == "complete"
    assert sum(1 for _ in (tmp_path / "run" / "frames.csv").open()) > 12000


def test_bad_row_does_not_discard_other_ztf_rows(tmp_path):
    text = (FIXTURES / "ztf.tbl").read_text()
    text = text.replace("757447121115", "        null")
    result = run(validate(FIELD), tmp_path, transport_factory=transport(text))
    assert len(result["frames"]) == 5
    assert result["checks"][0]["receivedRows"] == 6
    assert result["checks"][0]["rejectedRows"] == 1
    assert result["checks"][0]["status"] == "partial"


def test_resume_skips_successful_queries_and_preserves_attempts(tmp_path):
    x = validate(FIELD)
    success = run(x, tmp_path, transport_factory=transport((FIXTURES / "ztf.tbl").read_text()))
    resumed = run(x, tmp_path, resume=True, transport_factory=transport("", RuntimeError("must not run")))
    assert success["frames"] == resumed["frames"]
    assert len(list((tmp_path / "queries").glob("*/*/receipt.json"))) == 1
    with pytest.raises(ValueError):
        run(validate({**FIELD, "ra": 10}), tmp_path, resume=True)


def test_failed_retry_keeps_prior_valid_rows(tmp_path):
    text = (FIXTURES / "ztf.tbl").read_text().replace("757447121115", "        null")
    run(validate(FIELD), tmp_path, transport_factory=transport(text))
    result = run(validate(FIELD), tmp_path, resume=True, retry_partial=True, transport_factory=transport("", RuntimeError("offline")))
    assert len(result["frames"]) == 5
    assert result["checks"][0]["status"] == "error"
    assert len(list((tmp_path / "queries").glob("*/*/receipt.json"))) == 2


def test_cometary_and_parabolic_most_parameters():
    comet = {k: v for k, v in ORBIT.items() if k not in {"a", "meanAnomaly"}}
    comet.update(kind="cometary", q=1.36, tpJd=2460978.5, e=6.1)
    x = validate({"mode": "orbits", "start": "2025-01-01", "end": "2026-01-01", "providers": ["most:wise_merge"], "orbits": [comet]})
    p = most_params(plan(x)[0][0], x)
    assert p["obj_type"] == "Comet" and p["perih_time"] == 2460978.5 and p["perih_dist"] == 1.36
    assert "semimajor_axis" not in p
    assert ssois_params(Job("ssois", "nominal", orbit=x["orbits"][0]), x)["a"] < 0
    x["orbits"][0]["e"] = 1
    assert most_params(plan(x)[0][0], x)["eccentricity"] == 1


def test_ephemeris_wrap_poles_sigma_and_no_extrapolation():
    points = [{"mjd": 60000, "ra": 359, "dec": 0, "sigmaArcsec": 10},
              {"mjd": 60002, "ra": 1, "dec": 0, "sigmaArcsec": 30}]
    p = interpolate(points, 60001)
    assert distance(p["ra"], p["dec"], 0, 0) < 1e-8
    assert p["sigmaArcsec"] == 20
    assert interpolate(points, 59999) is None
    assert polygon_intersects(0, 0, .01, [(359, -1), (1, -1), (1, 1), (359, 1)])
    assert not polygon_intersects(180, 0, .01, [(359, -1), (1, -1), (1, 1), (359, 1)])
    assert distance(0, 90, 180, 90) < 1e-8


def test_ephemeris_readers_and_observer_required():
    rows = read_ephemeris("date,ra,dec,sigmaArcsec\n2023-02-25,359,0,0.51d\n2023-02-26,1,0,10m\n")
    assert rows[0]["sigmaArcsec"] == 1836
    assert rows[1]["sigmaArcsec"] == .01
    assert read_ephemeris("mjd,ra,dec\n60000.5,1,2\n60001.5,2,3\n")[0]["mjd"] == 60000.5
    x = {**FIELD, "mode": "ephemeris", "ephemeris": rows}
    with pytest.raises(ValueError, match="observer"):
        validate(x)
    parsed = read_ephemeris("2023 02 25  23 59 00.0 +00 01 00.0  1.0 2.0 3.0 4.0 500m 90\n2023 02 26  00 01 00.0 +00 01 00.0  1.0 2.0 3.0 4.0 1d 90\n")
    assert parsed[0]["sigmaArcsec"] == .5
    assert parsed[1]["sigmaArcsec"] == 3600


def test_credential_configuration_is_private_and_redacted(tmp_path, monkeypatch):
    p = tmp_path / "credentials.json"
    initialize(p)
    monkeypatch.setenv("HSC_SSP_CAS_USER", "account")
    monkeypatch.setenv("HSC_SSP_CAS_PASSWORD", "secret-fixture")
    c = Credentials(p)
    assert c.status()["hsc"]["configured"]
    assert "secret-fixture" not in c.redact("error secret-fixture")
    assert "secret-fixture" not in p.read_text()
    p.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        Credentials(p)


def test_missing_hsc_credentials_is_blocked_not_empty_coverage(tmp_path, monkeypatch):
    monkeypatch.delenv("HSC_SSP_CAS_USER", raising=False)
    monkeypatch.delenv("HSC_SSP_CAS_PASSWORD", raising=False)
    result = run(validate({**FIELD, "providers": ["hsc"]}), tmp_path, credentials=Credentials(tmp_path / "absent"), transport_factory=transport(""))
    assert result["checks"][0]["status"] == "blocked"
    assert result["status"] == "incomplete"


def test_cross_adapter_native_identity_and_product_provenance(tmp_path):
    result = run(validate(FIELD), tmp_path, transport_factory=transport((FIXTURES / "ztf.tbl").read_text()))
    other = json.loads(json.dumps(result))
    for f in other["frames"]:
        f["provider"] = "ssois"
        f["id"] = Path(urlsplit(f["url"]).path).name
        f["instrument"] = "Palomar/ZTF"
        f["hypotheses"] = ["orbit-1"]
        f["provenance"] = [{"provider": "ssois", "jobId": "other", "hypothesis": "orbit-1"}]
        f["key"] = canonical(f)
    merged = merge_manifests([result, other])
    assert len(merged["frames"]) == 6
    assert merged["frames"][0]["hypotheses"] == ["field", "orbit-1"]
    assert len(merged["frames"][0]["provenance"]) == 2


def test_mpc80_arc_wire_protocol():
    x = validate({**FIELD, "mode": "arc", "providers": ["ssois"], "arc": "x"*80+"\n"+"y"*80+"\n"})
    p = ssois_params(plan(x)[0][0], x)
    assert p["search"] == "bern" and len(p["obs"].splitlines()[1]) == 80
    with pytest.raises(ValueError, match="at least two"):
        validate({**x, "arc": " "*80+"\n"+"x"*80+"\n"})


def test_capabilities_and_invalid_cli_are_machine_readable(capsys, tmp_path):
    assert main(["capabilities"]) == 0
    assert json.loads(capsys.readouterr().out)["clientRowLimit"] is None
    p = tmp_path / "bad.json"
    p.write_text("{}")
    assert main(["search", str(p), "--run-dir", str(tmp_path / "run")]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "invalid"


@pytest.mark.parametrize("provider,fixture,ra,dec,start,end,count", [
    ("smoka", "smoka.txt", 150, 2, "2015-01-16", "2015-01-17", 4),
    ("noirlab", "noirlab.json", 150, 2, "2019-02-02", "2019-02-03", 3),
    ("mast", "mast.json", 150, 2, "2010-01-01", "2020-01-01", 2),
    ("catch", "catch.json", 180, 0, "2020-01-01", "2021-01-01", 2),
])
def test_live_deep_archive_fixtures(tmp_path, provider, fixture, ra, dec, start, end, count):
    def response(url, kwargs):
        if provider == "smoka" and "instruments=SUP" in url:
            return "No matching frame"
        return (FIXTURES / fixture).read_text()
    result = run(validate({**FIELD, "providers": [provider], "ra": ra, "dec": dec, "start": start, "end": end}),
                 tmp_path, transport_factory=transport(response))
    assert result["status"] == "complete"
    assert len(result["frames"]) == count
    assert all(f["url"] for f in result["frames"])
    if provider == "noirlab":
        f = result["frames"][0]
        assert f["startMjd"] + f["exposureSeconds"]/172800 == pytest.approx(mjd(f["raw"]["dateobs_center"]))
    if provider == "mast":
        assert any(f["linkKind"] == "product manifest" for f in result["frames"])


def test_most_wise_retains_both_bands(tmp_path):
    x = validate({"mode": "object", "object": "Victoria", "start": "2014-05-01", "end": "2014-05-30", "providers": ["most:wise_merge"]})
    result = run(x, tmp_path, transport_factory=transport((FIXTURES / "wise.tbl").read_text()))
    assert result["checks"][0]["receivedRows"] == 12
    assert len(result["frames"]) == 12
    assert {f["filter"] for f in result["frames"]} == {"1", "2"}


def test_noirlab_paginates_past_first_thousand(tmp_path):
    original = json.loads((FIXTURES / "noirlab.json").read_text())
    offsets = []
    def response(url, kwargs):
        offset = int(parse_qs(urlsplit(url).query)["offset"][0])
        offsets.append(offset)
        rows = [{**original[1], "archive_filename": f"frame-{i}.fits.fz"} for i in range(offset, min(offset+1000, 1003))]
        return json.dumps([original[0], *rows])
    result = run(validate({**FIELD, "ra": 150, "dec": 2, "start": "2019-02-02", "end": "2019-02-03", "providers": ["noirlab"]}),
                 tmp_path, transport_factory=transport(response))
    assert offsets == [0, 1000] and len(result["frames"]) == 1003


def test_server_overflow_creates_durable_child_queries(tmp_path):
    requests = []
    def response(url, kwargs):
        requests.append(url)
        state = "OVERFLOW" if len(requests) == 1 else "OK"
        return f'<VOTABLE><RESOURCE><TABLE><FIELD name="obs_id"/><DATA><TABLEDATA/></DATA></TABLE><INFO name="QUERY_STATUS" value="{state}"/></RESOURCE></VOTABLE>'
    x = validate({**FIELD, "start": "2019-01-01", "end": "2019-01-05", "providers": ["eso"]})
    result = run(x, tmp_path, transport_factory=transport(response))
    assert [c["status"] for c in result["checks"]] == ["split", "ok", "ok"]
    assert result["status"] == "complete"
    resumed = run(x, tmp_path, resume=True, transport_factory=transport("", RuntimeError("must not run")))
    assert len(resumed["checks"]) == 3


def test_moving_match_uses_exposure_midpoint_and_per_date_sigma():
    points = [{"mjd": 60000, "ra": 359, "dec": 0, "sigmaArcsec": 0},
              {"mjd": 60002, "ra": 1, "dec": 0, "sigmaArcsec": 720}]
    w = {"start": 60000, "end": 60002, "points": points, "label": "track"}
    x = {"radius": 0, "sigmaMultiplier": 1, "eellipse": 0, "observer": "T09"}
    f = {"startMjd": 60001-30/86400, "exposureSeconds": 60, "timeScale": "UTC",
         "footprint": "CIRCLE ICRS 0.15 0 0.06"}
    assert select(f, w, x)
    assert f["predictions"][0]["sigmaArcsec"] == pytest.approx(360)
    assert f["predictions"][0]["mjdUtc"] == pytest.approx(60001)
    assert not select({**f, "footprint": "CIRCLE ICRS 5 0 0.06"}, w, x)


def test_auth_is_removed_on_cross_origin_redirects():
    from urllib.request import Request
    from magictools.frames.transport import SameOriginRedirect
    handler = SameOriginRedirect()
    req = Request("https://mast.stsci.edu/a", headers={"Authorization": "token fixture"})
    for destination in ["https://other.example/b", "http://mast.stsci.edu/b"]:
        redirected = handler.redirect_request(req, None, 302, "redirect", {}, destination)
        assert not redirected.has_header("Authorization")
    same = handler.redirect_request(req, None, 302, "redirect", {}, "https://mast.stsci.edu/b")
    assert same.get_header("Authorization") == "token fixture"


def test_hsc_uses_published_frame_columns_and_keeps_credentials_private(tmp_path, monkeypatch):
    import re
    monkeypatch.setenv("HSC_SSP_CAS_USER", "account")
    monkeypatch.setenv("HSC_SSP_CAS_PASSWORD", "hsc-fixture-secret")
    columns = {row[1] for row in json.loads((FIXTURES / "hsc-schema.json").read_text())["columns"]}
    def response(url, kwargs):
        assert kwargs["private_data"]["credential"]["password"] == "hsc-fixture-secret"
        assert "credential" not in kwargs["data"]
        if url.endswith("submit"):
            sql = kwargs["data"]["catalog_job"]["sql"]
            assert set(re.search(r"SELECT (.*?) FROM", sql)[1].split(",")) <= columns
            return '{"id": 123}'
        if url.endswith("status"):
            return '{"status": "done"}'
        return "# metadata\nframe_id,utcobs\n"
    result = run(validate({**FIELD, "providers": ["hsc"]}), tmp_path,
                 credentials=Credentials(tmp_path / "absent"), transport_factory=transport(response))
    assert result["status"] == "complete"
    assert "hsc-fixture-secret" not in json.dumps(result)
    assert "hsc-fixture-secret" not in (tmp_path / "manifest.json").read_text()


def test_ephemeris_missing_dates_are_explicit_and_combined_products_remain_candidates():
    x = validate({**FIELD, "mode": "ephemeris", "start": "2023-02-24", "end": "2023-02-28", "observer": "T09",
                  "ephemeris": [{"mjd": 60000, "ra": 0, "dec": 0}, {"mjd": 60001, "ra": 1, "dec": 0}]})
    _, unavailable = plan(x)
    assert any("outside this interval are unqueried" in u["reason"] for u in unavailable)
    f = {"startMjd": 59000, "endMjd": 61000, "timeScale": "UTC"}
    assert select(f, {"start": 60000, "end": 60001, "points": x["ephemeris"]}, x)
    assert "combined product" in f["match"]


def test_local_background_api_and_disk_status(tmp_path, monkeypatch):
    import os
    import socket
    import subprocess
    import sys
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.delenv("HSC_SSP_CAS_USER", raising=False)
    monkeypatch.delenv("HSC_SSP_CAS_PASSWORD", raising=False)
    process = subprocess.Popen([sys.executable, "-m", "magictools.frames.cli", "serve", "--port", str(port),
                                "--directory", str(tmp_path / "runs"), "--credentials", str(tmp_path / "absent")],
                               env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with urlopen(base+"/api/frames", timeout=1) as response:
                    assert json.load(response)["clientRowLimit"] is None
                break
            except URLError:
                time.sleep(.05)
        else:
            pytest.fail("Local API did not start")
        body = json.dumps({**FIELD, "providers": ["hsc"]}).encode()
        req = Request(base+"/api/frames", data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=2) as response:
            assert response.status == 202
            job = json.load(response)
        for _ in range(100):
            with urlopen(base+job["pollUrl"], timeout=2) as response:
                state = json.load(response)
            if state.get("jobStatus") == "finished":
                break
            time.sleep(.05)
        assert state["status"] == "incomplete" and state["checks"][0]["status"] == "blocked"
        req.add_header("Origin", "https://example.com")
        with pytest.raises(HTTPError) as rejected:
            urlopen(req, timeout=2)
        assert rejected.value.code == 400
    finally:
        process.terminate()
        process.communicate(timeout=5)
    saved = status(tmp_path / "runs" / job["id"])
    assert saved["checks"][0]["status"] == "blocked"
