"""Progress must appear during blocked requests without contaminating results."""
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from magictools.frames import cli, engine
from magictools.frames.adapters import frame
from magictools.frames.model import validate
from magictools.frames.progress import TextProgress

FIELD = {"mode": "field", "ra": 150, "dec": 2, "radius": .01,
         "start": "2020-01-01", "end": "2020-01-03", "providers": ["ztf"]}


def test_progress_during_response_wait_download_and_candidate_processing(tmp_path, monkeypatch):
    allow_headers, allow_finish, allow_rows = (threading.Event() for _ in range(3))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            allow_headers.wait(3)
            self.send_response(200)
            self.send_header("Content-Length", "65537")
            self.end_headers()
            self.wfile.write(b"x" * 65536)
            self.wfile.flush()
            allow_finish.wait(3)
            self.wfile.write(b"x")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events, output = [], io.StringIO()
    render = TextProgress(output)

    def execute(job, value, transport, counters):
        text = transport.request(f"http://127.0.0.1:{server.server_port}/?secret=not-for-progress")
        assert len(text) == 65537
        yield frame(job, {}, id="candidate-1")
        allow_rows.wait(3)

    def emit(event):
        events.append(event)
        render(event)
        if event["type"] == "progress":
            active = event["active"][0]
            if active["phase"] == "waiting for archive response":
                allow_headers.set()
            if active["responseBytes"] >= 65536:
                allow_finish.set()
            if active["candidateRows"] == 1:
                allow_rows.set()

    monkeypatch.setattr(engine, "execute", execute)
    try:
        result = engine.run(validate(FIELD), tmp_path, emit=emit, progress_interval=.01)
    finally:
        allow_headers.set()
        allow_finish.set()
        allow_rows.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    snapshots = [e for e in events if e["type"] == "progress"]
    assert all(e["finishedQueries"] == 0 and e["totalQueries"] == 1 for e in snapshots)
    assert all(e["active"][0]["window"] == {"start": 58849, "end": 58851} for e in snapshots)
    assert any(e["active"][0]["phase"] == "waiting for archive response" for e in snapshots)
    assert any(e["active"][0]["responseBytes"] >= 65536 for e in snapshots)
    assert any(e["candidateRowsThisRun"] == 1 for e in snapshots)
    assert "not-for-progress" not in json.dumps(snapshots)
    assert "not-for-progress" not in output.getvalue()
    assert "waiting for archive response" in output.getvalue()
    assert "1/1" in output.getvalue() and "1 deduplicated candidate frames" in output.getvalue()
    assert events[-1]["type"] == "complete" and result["status"] == "complete"


def test_split_totals_and_resume_checkpoints_are_reported(tmp_path, monkeypatch):
    def execute(job, value, transport, counters):
        if job.window["end"] - job.window["start"] > 1:
            counters["incomplete"] = "Service overflow"
        else:
            yield frame(job, {}, id="same-exposure")

    monkeypatch.setattr(engine, "execute", execute)
    events = []
    result = engine.run(validate(FIELD), tmp_path, emit=events.append)
    done = [e for e in events if e["type"] == "query-complete"]
    assert [(e["finishedQueries"], e["totalQueries"]) for e in done] == [(1, 3), (2, 3), (3, 3)]
    assert len(result["frames"]) == 1  # Two rows are one exposure after deduplication.
    events.clear()
    monkeypatch.setattr(engine, "execute", lambda *_: pytest.fail("Successful checkpoints must be reused"))
    resumed = engine.run(validate(FIELD), tmp_path, resume=True, emit=events.append, progress_interval=2)
    assert events[0]["reusedQueries"] == 3 and events[0]["queries"] == 0
    assert resumed["frames"] == result["frames"]
    assert [e["type"] for e in events] == ["plan", "finalizing", "complete"]


@pytest.mark.parametrize("output_format", ["json", "csv", "ndjson"])
@pytest.mark.parametrize("progress", ["text", "json", "none"])
def test_cli_progress_keeps_output_contracts(output_format, progress, tmp_path, monkeypatch, capsys):
    def execute(job, value, transport, counters):
        yield frame(job, {}, id="test-exposure")

    monkeypatch.setattr(engine, "execute", execute)
    path = tmp_path / "query.json"
    path.write_text(json.dumps(FIELD))
    code = cli.main(["search", str(path), "--run-dir", str(tmp_path / "run"),
                     "--format", output_format, "--progress", progress])
    out, err = capsys.readouterr()
    assert code == 0
    if output_format == "json":
        assert json.loads(out)["frames"][0]["id"] == "test-exposure"
    elif output_format == "csv":
        import csv
        assert list(csv.DictReader(io.StringIO(out)))[0]["id"] == "test-exposure"
    else:
        events = [json.loads(line) for line in out.splitlines()]
        assert events[0]["type"] == "plan" and events[-1]["type"] == "complete"
    if progress == "text":
        assert "START ztf" in err and "1/1" in err and "COMPLETE" in err
    elif progress == "json":
        assert [json.loads(line)["type"] for line in err.splitlines()][-1] == "complete"
    else:
        assert err == ""


def test_blocked_and_unqueried_are_visible(tmp_path, monkeypatch):
    from magictools.frames.transport import Blocked

    def execute(*_):
        raise Blocked("Archive requires credentials")

    monkeypatch.setattr(engine, "execute", execute)
    output = io.StringIO()
    result = engine.run(validate({**FIELD, "providers": ["ztf", "ssois"]}), tmp_path, emit=TextProgress(output))
    assert result["status"] == "incomplete"
    assert "UNQUERIED ssois" in output.getvalue()
    assert "BLOCKED" in output.getvalue() and "Archive requires credentials" in output.getvalue()
    assert "INCOMPLETE" in output.getvalue()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_progress_interval_is_validated_before_starting(interval, tmp_path):
    with pytest.raises(ValueError, match="progress interval"):
        engine.run(validate(FIELD), tmp_path / "new", progress_interval=interval)
    assert not (tmp_path / "new").exists()
