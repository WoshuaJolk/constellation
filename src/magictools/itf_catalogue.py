"""Whole-file ITF preparation. Every input line survives, regardless of parse status."""
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import urllib.request

import numpy as np

MPC_ITF_URL = "https://www.minorplanetcenter.net/iau/ITF/itf.txt.gz"
FORMAT_URL = "https://docs.minorplanetcenter.net/mpc-ops-docs/observations/mpc1992-format/"
DTYPE = np.dtype([("designation", "S12"), ("station", "S3"), ("note2", "S1"),
                  ("status", "u1"), ("mjd_utc", "<f8"), ("ra_deg", "<f8"), ("dec_deg", "<f8")])
STATUS = {0: "optical_fields_parsed", 1: "auxiliary_record",
          2: "unsupported_observation_type", 3: "malformed_record",
          4: "optical_needing_observer_context", 5: "suppressed_historical_record"}
MJD_ZERO = date(1858, 11, 17).toordinal()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def fetch(directory, url=MPC_ITF_URL):
    """Fetch the complete response once; reject short or non-gzip responses.

    Gzip integrity is checked during prepare. A receipt does not assert that the
    upstream file is an authoritative census beyond this particular snapshot.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"schema": "solve-itf-download-v1", "requested_url": url,
               "retrieved_utc": datetime.now(timezone.utc).isoformat(), "complete": False}
    partial = directory/"source.gz.part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "constellation-solve-itf/0.1",
                                                       "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as out:
            receipt.update(resolved_url=response.geturl(),
                           content_length=response.headers.get("Content-Length"),
                           etag=response.headers.get("ETag"),
                           last_modified=response.headers.get("Last-Modified"))
            if response.status != 200:
                raise ValueError("Expected a complete HTTP 200 response")
            for chunk in iter(lambda: response.read(1024*1024), b""):
                out.write(chunk)
        size = partial.stat().st_size
        if receipt["content_length"] is not None and size != int(receipt["content_length"]):
            raise ValueError("Truncated HTTP response")
        with partial.open("rb") as source:
            if source.read(2) != b"\x1f\x8b":
                raise ValueError("Expected a gzip archive, not an HTML error page")
        partial.rename(directory/"source.gz")
        receipt.update(complete=True, bytes=size, sha256=digest(directory/"source.gz"))
        return receipt
    except BaseException as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["download_seconds"] = time.monotonic()-started
        save(directory/"download.json", receipt)


def parse_record(line):
    """Parse optical fields without deciding source identity or creating priors.

    Dates remain UTC; J2000 coordinate fields are not relabelled ICRS or corrected
    for aberration. Space/roving observers and historical records remain flagged.
    """
    raw = line.rstrip(b"\r\n")
    designation, station, note = raw[:12], raw[77:80], raw[14:15]
    empty = (designation, station, note, 3, np.nan, np.nan, np.nan)
    if len(raw) != 80 or any(c < 32 or c > 126 for c in raw):
        return empty
    if note in (b"s", b"v", b"w", b"t", b"r", b"q"):
        return (designation, station, note, 1, np.nan, np.nan, np.nan)
    if note not in (b" ", b"A", b"P", b"C", b"B", b"c", b"D", b"Z", b"e",
                    b"M", b"E", b"H", b"N", b"n", b"S", b"V", b"W", b"T", b"X", b"x"):
        return (designation, station, note, 2, np.nan, np.nan, np.nan)
    try:
        year, month, day = raw[15:32].split()
        day = float(day)
        integer_day = int(day)
        ordinal = date(int(year), int(month), integer_day).toordinal()
        if not integer_day <= day < integer_day+1:
            return empty
        hours, minutes, seconds = map(float, raw[32:44].split())
        sign = raw[44:45]
        degrees, arcmin, arcsec = map(float, raw[45:56].split())
        if (sign not in (b"+", b"-") or not 0 <= hours < 24
                or not 0 <= minutes < 60 or not 0 <= seconds < 60
                or not 0 <= degrees <= 90 or not 0 <= arcmin < 60 or not 0 <= arcsec < 60
                or not station.strip() or not designation.strip()):
            return empty
        ra = 15*(hours+minutes/60+seconds/3600)
        dec = degrees+arcmin/60+arcsec/3600
        if dec > 90:
            return empty
        if sign == b"-":
            dec = -dec
        status = 4 if note in (b"S", b"V", b"W", b"T") else 5 if note in (b"X", b"x") else 0
        return (designation, station, note, status, ordinal-MJD_ZERO+day-integer_day, ra, dec)
    except (ValueError, OverflowError):
        return empty


def _update_index(connection, records):
    """Index label/station groups; these are NOT certified physical tracklets."""
    keys = np.empty(len(records), dtype=[("designation", "S12"), ("station", "S3")])
    keys["designation"], keys["station"] = records["designation"], records["station"]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    parsed = np.isfinite(records["mjd_utc"])
    parsed_counts = np.bincount(inverse[parsed], minlength=len(unique))
    lo, hi = np.full(len(unique), np.inf), np.full(len(unique), -np.inf)
    np.minimum.at(lo, inverse[parsed], records["mjd_utc"][parsed])
    np.maximum.at(hi, inverse[parsed], records["mjd_utc"][parsed])
    values = [(bytes(k["designation"]), bytes(k["station"]), int(c), int(p),
               None if p == 0 else float(a), None if p == 0 else float(b))
              for k, c, p, a, b in zip(unique, counts, parsed_counts, lo, hi)]
    connection.executemany(
        """INSERT INTO label_groups VALUES(?,?,?,?,?,?)
        ON CONFLICT(designation,station) DO UPDATE SET
        records=records+excluded.records, parsed_records=parsed_records+excluded.parsed_records,
        first_mjd=CASE WHEN first_mjd IS NULL THEN excluded.first_mjd
                      WHEN excluded.first_mjd IS NULL THEN first_mjd
                      ELSE min(first_mjd,excluded.first_mjd) END,
        last_mjd=CASE WHEN last_mjd IS NULL THEN excluded.last_mjd
                     WHEN excluded.last_mjd IS NULL THEN last_mjd
                     ELSE max(last_mjd,excluded.last_mjd) END""", values)
    connection.commit()


def prepare(source, directory, chunk_rows=200000):
    """Full streaming ingestion, no sample/date/station/row-count cutoff."""
    source, directory = Path(source).resolve(), Path(directory)
    if not isinstance(chunk_rows, int) or not 1 <= chunk_rows <= 1000000:
        raise ValueError("chunk_rows must be between 1 and 1000000 (batch size, not an input limit)")
    if not source.is_file():
        raise ValueError("Source file does not exist")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"schema": "solve-itf-catalogue-v1", "status": "building",
                "source_path": str(source), "all_input_records_retained": False,
                "full_itf_search_complete": False, "orbit_search_attempted": False,
                "grouping": "Exact source designation bytes + station; not physical tracklet identity",
                "format_reference": FORMAT_URL, "coordinate_fields": "MPC J2000 as supplied",
                "time_scale": "UTC as supplied; no TDB or observer-state conversion",
                "record_dtype": DTYPE.descr, "record_bytes": DTYPE.itemsize,
                "shards": [], "status_codes": STATUS}
    connection = None
    try:
        with source.open("rb") as inp:
            compressed = inp.read(2) == b"\x1f\x8b"
        raw_name = "source.gz" if compressed else "source.obs80"
        shutil.copyfile(source, directory/raw_name)
        raw_hash = digest(directory/raw_name)
        manifest.update(raw_file=raw_name, source_sha256=raw_hash, source_bytes=source.stat().st_size)
        receipt = source.parent/"download.json"
        if receipt.is_file():
            candidate = json.loads(receipt.read_text())
            if candidate.get("complete") and candidate.get("sha256") == raw_hash:
                manifest["download_receipt"] = candidate
        connection = sqlite3.connect(directory/"labels.sqlite")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE label_groups(designation BLOB,station BLOB,records INTEGER,"
                           "parsed_records INTEGER,first_mjd REAL,last_mjd REAL,"
                           "PRIMARY KEY(designation,station)) WITHOUT ROWID")
        block = np.empty(chunk_rows, dtype=DTYPE)
        count, used, decoded_bytes = 0, 0, 0
        decoded_hash = hashlib.sha256()
        statuses = Counter()
        samples = []
        def flush():
            nonlocal used
            if not used:
                return
            records = block[:used]
            name = f"records-{len(manifest['shards']):05d}.npy"
            np.save(directory/name, records, allow_pickle=False)
            _update_index(connection, records)
            manifest["shards"].append({"file": name, "first_line": count-used+1, "rows": used,
                                      "bytes": (directory/name).stat().st_size,
                                      "sha256": digest(directory/name)})
            used = 0
            print(json.dumps({"event": "itf_records_ingested", "records": count}), file=sys.stderr, flush=True)
        opener = gzip.open if compressed else open
        with opener(directory/raw_name, "rb") as inp:
            for line in inp:
                decoded_hash.update(line)
                decoded_bytes += len(line)
                record = parse_record(line)
                block[used] = record
                statuses[STATUS[record[3]]] += 1
                count += 1
                used += 1
                if record[3] and len(samples) < 20:
                    samples.append({"line": count, "status": STATUS[record[3]]})
                if used == chunk_rows:
                    flush()
            flush()  # Reaching EOF also checks gzip CRC/trailer integrity.
        if not count:
            raise ValueError("Empty source archive")
        manifest.update(records=count, decoded_bytes=decoded_bytes,
                        decoded_sha256=decoded_hash.hexdigest(), status_counts=dict(statuses),
                        flagged_record_examples=samples,
                        label_station_groups=connection.execute("SELECT count(*) FROM label_groups").fetchone()[0],
                        multi_day_label_groups=connection.execute(
                            "SELECT count(*) FROM label_groups WHERE last_mjd-first_mjd>1").fetchone()[0],
                        index_records=connection.execute("SELECT sum(records) FROM label_groups").fetchone()[0])
        connection.close()
        connection = None
        manifest.update(index_sha256=digest(directory/"labels.sqlite"),
                        hot_column_bytes=sum(s["bytes"] for s in manifest["shards"]),
                        index_bytes=(directory/"labels.sqlite").stat().st_size,
                        status="prepared", all_input_records_retained=True,
                        astrometry_ready_for_orbit_inference=False,
                        unresolved_preparation=["observer states", "UTC-to-TDB conversion",
                                                "uncertainty/calibration model", "tracklet identity",
                                                "flagged/auxiliary record interpretation"])
        if manifest["records"] != manifest["index_records"] or sum(statuses.values()) != count:
            raise ArithmeticError("Record accounting mismatch")
        return manifest
    except BaseException as error:
        manifest.update(status="failed", all_input_records_retained=False, error=str(error))
        raise
    finally:
        if connection is not None:
            connection.close()
        manifest["preparation_seconds"] = time.monotonic()-started
        save(directory/"catalogue.json", manifest)


def verify(directory):
    """Read every shard and every original line; reject corruption or missing coverage."""
    directory = Path(directory)
    manifest = json.loads((directory/"catalogue.json").read_text())
    if manifest.get("schema") != "solve-itf-catalogue-v1" or manifest.get("status") != "prepared":
        raise ValueError("Not a completed catalogue")
    def local(name):
        path = (directory/name).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Invalid catalogue file path")
        return path
    raw = local(manifest["raw_file"])
    if digest(raw) != manifest["source_sha256"]:
        raise ValueError("Original source checksum mismatch")
    decoded = hashlib.sha256()
    count, size = 0, 0
    with raw.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(raw, "rb") as stream:
        for line in stream:
            decoded.update(line)
            size += len(line)
            count += 1
    if (count != manifest["records"] or size != manifest["decoded_bytes"]
            or decoded.hexdigest() != manifest["decoded_sha256"]):
        raise ValueError("Raw line accounting mismatch")
    next_line, counts = 1, Counter()
    for shard in manifest["shards"]:
        path = local(shard["file"])
        if digest(path) != shard["sha256"] or shard["first_line"] != next_line:
            raise ValueError("Shard integrity or ordering mismatch")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.dtype != DTYPE or data.shape != (shard["rows"],):
            raise ValueError("Shard schema mismatch")
        values, frequencies = np.unique(data["status"], return_counts=True)
        for code, frequency in zip(values, frequencies):
            if int(code) not in STATUS:
                raise ValueError("Unknown record status")
            counts[STATUS[int(code)]] += int(frequency)
        next_line += len(data)
    if next_line-1 != count or dict(counts) != manifest["status_counts"]:
        raise ValueError("Column record accounting mismatch")
    index = local("labels.sqlite")
    if digest(index) != manifest["index_sha256"]:
        raise ValueError("Index checksum mismatch")
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Index integrity failure")
        rows = connection.execute("SELECT sum(records) FROM label_groups").fetchone()[0]
    if rows != count:
        raise ValueError("Index line accounting mismatch")
    return {"verified": True, "all_input_records_retained": True,
            "records": count, "source_sha256": manifest["source_sha256"],
            "full_itf_search_complete": False,
            "scope": "Checksums, EOF and exhaustive line/shard/index counts; not orbit-search coverage"}


def main(argv=None, prog="constellation solve_itf catalogue"):
    import argparse
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="Download the complete public MPC ITF snapshot")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare", help="Stream every input record into losslessly backed numeric shards")
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=200000)
    p = sub.add_parser("verify", help="Check every source record and shard")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("inspect", help="Read the preparation receipt")
    p.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch(args.output)
        elif args.command == "prepare":
            result = prepare(args.source, args.output, args.chunk_rows)
        elif args.command == "verify":
            result = verify(args.directory)
        else:
            result = json.loads((args.directory/"catalogue.json").read_text())
        print(json.dumps({"ok": True, "tool": "solve_itf", "operation": args.command,
                          "result": result}, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"ok": False, "tool": "solve_itf", "operation": args.command,
                          "error": str(error) or type(error).__name__}))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
