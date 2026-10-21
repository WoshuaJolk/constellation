"""Thread-safe query counters and readable CLI progress (never image coverage)."""
import threading
import time


class QueryProgress:
    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.values = {"phase": "starting", "candidateRows": 0, "requestNumber": 0, "responseBytes": 0}

    def update(self, **values):
        with self.lock:
            self.values.update(values)

    def snapshot(self):
        with self.lock:
            return {**self.values, "elapsedSeconds": round(time.monotonic() - self.started, 1)}


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def safe_text(value):
    return "".join(c if c.isprintable() else " " for c in str(value))


def query_context(job):
    return {"jobId": job.id, "provider": job.provider, "hypothesis": job.hypothesis,
            "instrument": job.params.get("instrument"),
            "window": {k: job.window[k] for k in ("start", "end")} if job.window else None}


def query_name(event):
    parts = [event["provider"], event["hypothesis"]]
    if event.get("instrument"):
        parts.append(str(event["instrument"]))
    if event.get("window"):
        window = event["window"]
        parts.append(f"MJD {window['start']:g}–{window['end']:g}")
    return " | ".join(parts)


class TextProgress:
    def __init__(self, stream):
        self.stream = stream
        self.started = time.monotonic()

    def write(self, text):
        self.stream.write(f"[{duration(time.monotonic() - self.started)}] {safe_text(text)}\n")
        self.stream.flush()

    def __call__(self, event):
        kind = event["type"]
        if kind == "plan":
            self.write(f"Search: {event['queries']} queries to run; {event.get('reusedQueries', 0)} reused from checkpoints. Run: {event['runDirectory']}")
            for item in event["unqueried"]:
                self.write(f"UNQUERIED {item.get('provider', item.get('name', 'coverage'))}: {item['reason']}")
        elif kind == "query-start":
            self.write(f"START {query_name(event)}")
        elif kind == "progress":
            self.write(f"{event['finishedQueries']}/{event['totalQueries']} queries finished; {len(event['active'])} active, {event['queuedQueries']} queued; {event['candidateRowsThisRun']} candidate rows this run (before deduplication)")
            for job in event["active"]:
                detail = f"request {job['requestNumber']}, {job['responseBytes']:,} bytes, " if job["requestNumber"] else ""
                self.write(f"  {query_name(job)} | {job['phase']} | {detail}{job['candidateRows']} candidate rows | {duration(job['elapsedSeconds'])} elapsed")
        elif kind == "query-complete":
            self.write(f"{event['finishedQueries']}/{event['totalQueries']} {query_name(event)} | {event['status'].upper()}: {event['count']} candidate rows; {event['rejectedRows']} rejected, {event['excludedRows']} excluded")
            if event["status"] != "ok":
                self.write(event["message"])
        elif kind == "finalizing":
            self.write("Finalizing: deduplicating frames and writing JSON/CSV artifacts")
        elif kind == "complete":
            self.write(f"{event['status'].upper()}: {event['count']} deduplicated candidate frames. Manifest: {event['manifest']}; CSV: {event['csv']}")
