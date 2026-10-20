"""Optional loopback-only background API, backed by the same durable CLI runs."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shlex
import uuid

from .engine import run, status
from .model import validate


def serve(directory, port, credentials):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    pool = ThreadPoolExecutor(max_workers=2)
    active = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log user bodies or credential-bearing requests.

        def send_json(self, value, code=200):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                self.send_json({"error": "Use the loopback address"}, 400)
                return
            if self.path == "/api/frames":
                from .cli import capabilities
                self.send_json(capabilities())
                return
            match = re.fullmatch(r"/api/frames/jobs/([a-f0-9]{32})", self.path)
            if not match or not (root / match[1] / "input.json").exists():
                self.send_json({"error": "Unknown job"}, 404)
                return
            job_id = match[1]
            future = active.get(job_id)
            if future and future.done() and future.exception():
                self.send_json({"id": job_id, "status": "error", "error": credentials.redact(str(future.exception()))}, 200)
                return
            try:
                result = status(root / job_id)
                result["jobStatus"] = "running" if future and not future.done() else "finished" if (root / job_id / "manifest.json").exists() else "interrupted"
                self.send_json(result)
            except (ValueError, OSError, KeyError):
                self.send_json({"id": job_id, "jobStatus": "queued"})

        def do_POST(self):
            if self.path != "/api/frames":
                self.send_json({"error": "Unknown route"}, 404)
                return
            try:
                if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                    raise ValueError("Use the loopback address")
                if self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
                    raise ValueError("Use an agent HTTP client with Content-Type: application/json; browser origins are not accepted")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Provide a JSON body no larger than 32 MiB")
                data = json.loads(self.rfile.read(length))
                # HTTP cannot read arbitrary local files. CLI manifests may do so.
                if "ephemerisFile" in data or "arcFile" in data:
                    raise ValueError("HTTP requests must inline ephemeris/arc data")
                x = validate(data)
                job_id = uuid.uuid4().hex
                folder = root / job_id
                folder.mkdir(mode=0o700)
                (folder / "input.json").write_text(json.dumps(x, allow_nan=False))
                active[job_id] = pool.submit(run, x, folder, credentials=credentials)
                self.send_json({"id": job_id, "status": "queued", "pollUrl": f"/api/frames/jobs/{job_id}",
                                "resumeCommand": f"constellation search {shlex.quote(str(folder / 'input.json'))} --run-dir {shlex.quote(str(folder))} --resume"}, 202)
            except (ValueError, TypeError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.shutdown(wait=False, cancel_futures=True)
