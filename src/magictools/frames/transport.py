"""Auditable downloads to disk, with timeouts and credentials kept out of receipts."""
import base64
import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_locks = {}
_guard = threading.Lock()


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        old_origin = urllib.parse.urlsplit(req.full_url)
        new_origin = urllib.parse.urlsplit(newurl)
        if new and (old_origin.scheme, old_origin.netloc) != (new_origin.scheme, new_origin.netloc):
            new.remove_header("Authorization")
            new.remove_header("Cookie")
            new.remove_header("X-api-key")
        return new


class ArchiveError(Exception):
    pass


class Blocked(ArchiveError):
    pass


class Transport:
    def __init__(self, directory, timeout=600, credentials=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.credentials = credentials
        self.calls = []

    def cached_request(self, url, *, binary=False):
        """Share immutable public metadata within a run, retaining a receipt per use."""
        root = Path(getattr(self, "shared_cache", self.directory / "metadata-cache"))
        root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode()).hexdigest()
        path, info = root / (key + ".response"), root / (key + ".json")
        with _guard:
            lock = _locks.setdefault("metadata:" + str(path), threading.Lock())
        with lock:
            if not path.exists() or not info.exists():
                downloaded = self.request(url, binary=True, auth={})
                if path.exists():
                    path.unlink()
                os.link(downloaded, path)
                info.write_text(json.dumps(self.calls[-1]))
            else:
                record = json.loads(info.read_text())
                # Hard links preserve raw artifacts for each attempt without copying
                # a 40 MB index hundreds of times along an ephemeris.
                local = self.directory / f"{len(self.calls):04d}-{key[:20]}.cached.response"
                os.link(path, local)
                record.update(cacheHit=True, responseFile=local.name, reusedAt=time.time())
                self.calls.append(record)
        if binary:
            return path
        payload = path.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        return payload.decode(errors="replace")

    def request(self, url, *, data=None, headers=None, auth=None, private_data=None, binary=False, sensitive_response=False):
        """`data` is recorded; `private_data` is transmitted but never recorded."""
        headers = dict(headers or {})
        host = urllib.parse.urlsplit(url).hostname
        profile = {"mast.stsci.edu": "mast", "astroarchive.noirlab.edu": "noirlab", "irsa.ipac.caltech.edu": "irsa", "archive-api.lco.global": "lco", "archive.gemini.edu": "gemini", "data.lsst.cloud": "rubin", "api.starglass.cfa.harvard.edu": "dasch", "splus.cloud": "splus"}.get(host)
        if auth is None and self.credentials and profile:
            auth = self.credentials.get(profile)
        public = {"url": url, "method": "POST" if data is not None or private_data is not None else "GET"}
        if data is not None:
            public["body"] = data
        body = private_data if private_data is not None else data
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            body = body.encode()
        headers.setdefault("User-Agent", "magictools-astro/frame-search (+https://github.com/WoshuaJolk/magictools)")
        if auth:
            if urllib.parse.urlsplit(url).scheme != "https":
                raise Blocked("Credentials require HTTPS; this archive endpoint is public HTTP only")
            if auth.get("cookie"):
                headers["Cookie"] = "gemini_archive_session=" + auth["cookie"]
            if auth.get("token") and profile == "dasch":
                headers["X-api-key"] = auth["token"]
            elif auth.get("token"):
                prefix = "token " if profile in {"mast", "lco"} else "" if profile == "noirlab" else "Bearer "
                headers["Authorization"] = prefix + auth["token"]
            elif auth.get("username") and auth.get("password"):
                encoded = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
        with _guard:
            lock = _locks.setdefault(host, threading.Lock())
        stamp = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()[:20]
        path = self.directory / f"{len(self.calls):04d}-{stamp}.response"
        record = {**public, "startedAt": time.time(), "timeoutSeconds": self.timeout}
        self.calls.append(record)
        digest = hashlib.sha256()
        size = 0
        with lock:
            try:
                request = urllib.request.Request(url, data=body, headers=headers)
                opener = urllib.request.build_opener(SameOriginRedirect())
                with opener.open(request, timeout=self.timeout) as response, path.with_suffix(".partial").open("wb") as out:
                    record["httpStatus"] = response.status
                    record["contentType"] = response.headers.get("Content-Type")
                    if sensitive_response:
                        # Login responses can contain a newly minted, unknown secret.
                        # Keep them in memory, never in raw artifacts or receipts.
                        payload = response.read(1048577)
                        path.with_suffix(".partial").unlink()
                        if len(payload) > 1048576:
                            raise ArchiveError("Authentication response exceeds 1 MiB")
                        record.update(responseRedacted=True, finishedAt=time.time())
                        return (gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload).decode()
                    deadline = time.monotonic() + self.timeout
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Response exceeded the requested transfer deadline")
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        out.write(chunk)
                path.with_suffix(".partial").replace(path)
                record.update(bytes=size, sha256=digest.hexdigest(), responseFile=path.name, finishedAt=time.time())
                if binary:
                    return path
                payload = path.read_bytes()
                if payload.startswith(b"\x1f\x8b"):
                    payload = gzip.decompress(payload)
                    record["responseEncoding"] = "gzip"
                text = payload.decode(errors="replace")
                redacted = self.credentials.redact(text) if self.credentials else text
                if redacted != text:
                    path.write_text(redacted)
                    record.update(responseRedacted=True, responseEncoding="identity", sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
                return redacted
            except urllib.error.HTTPError as e:
                record["httpStatus"] = e.code
                if e.code in (401, 403):
                    raise Blocked(f"Archive returned HTTP {e.code}; authentication or access is required") from None
                # Do not echo response bodies: some services echo credentials.
                raise ArchiveError(f"Archive returned HTTP {e.code}; inspect the recorded public request") from None
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                raise ArchiveError(f"Archive transport failed ({type(e).__name__}); retry or increase timeoutSeconds") from None
