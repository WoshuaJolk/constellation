"""Whole-input accounting and independent coordinate/calendar controls."""
import gzip
import json
import sqlite3

from astropy.time import Time
import numpy as np
import pytest

from magictools.itf_catalogue import DTYPE, parse_record, prepare, verify
from magictools import agent_cli


def record(label="ABC1234", station="F51", note="C", day="2024 02 29.250000"):
    row = [" "]*80
    row[5:12] = label
    row[14] = note
    row[15:32] = f"{day:<17}"
    row[32:44] = "23 59 59.900"
    row[44:56] = "-00 30 00.00"
    row[77:80] = station
    result = "".join(row).encode()
    assert len(result) == 80
    return result


def test_parse_coordinates_calendar_negative_zero_and_wrap():
    row = parse_record(record())
    assert row[3] == 0
    assert row[4] == pytest.approx(Time("2024-02-29T06:00:00", scale="utc").mjd, abs=1e-10)
    assert row[5] == pytest.approx(359.9995833333333)
    assert row[6] == -.5
    assert row[:3] == (b"     ABC1234", b"F51", b"C")


def test_every_line_survives_and_index_spans_shards(tmp_path):
    lines = [record(), record(station="G96")+b"\r",
             record(note="s"), b"malformed", b"", record(note="S"),
             record(note="X"), record(note="R"), record(day="2024 02 30.100000"),
             record(day="2024 03 02.100000")]
    raw = b"\n".join(lines)  # Last line deliberately lacks a terminator.
    source = tmp_path/"source.gz"
    source.write_bytes(gzip.compress(raw))
    out = tmp_path/"catalogue"
    report = prepare(source, out, chunk_rows=2)
    assert report["records"] == 10
    assert report["all_input_records_retained"]
    assert report["status_counts"] == {
        "optical_fields_parsed": 3, "auxiliary_record": 1, "malformed_record": 3,
        "optical_needing_observer_context": 1, "suppressed_historical_record": 1,
        "unsupported_observation_type": 1}
    assert gzip.decompress((out/"source.gz").read_bytes()) == raw
    assert (out/"source.gz").read_bytes() == source.read_bytes()
    rows = np.concatenate([np.load(out/s["file"], allow_pickle=False) for s in report["shards"]])
    assert len(rows) == len(lines) and rows.dtype == DTYPE
    assert rows["status"].tolist() == [0, 0, 1, 3, 3, 4, 5, 2, 3, 0]
    assert report["index_records"] == 10
    assert not report["astrometry_ready_for_orbit_inference"]
    assert not report["full_itf_search_complete"]
    assert verify(out)["verified"]
    with sqlite3.connect(out/"labels.sqlite") as conn:
        assert conn.execute("SELECT records FROM label_groups WHERE designation=? AND station=?",
                            (b"     ABC1234", b"F51")).fetchone()[0] == 7
    assert report["multi_day_label_groups"] == 1


@pytest.mark.parametrize("corruption", ["source", "shard", "index", "order"])
def test_verification_rejects_corruption(tmp_path, corruption):
    source = tmp_path/"source.txt"
    source.write_bytes(record()+b"\n"+record(station="G96"))
    out = tmp_path/"catalogue"
    report = prepare(source, out, chunk_rows=1)
    if corruption == "source":
        (out/"source.obs80").write_bytes(b"changed")
    elif corruption == "shard":
        with (out/report["shards"][0]["file"]).open("ab") as stream:
            stream.write(b"changed")
    elif corruption == "index":
        with (out/"labels.sqlite").open("ab") as stream:
            stream.write(b"changed")
    else:
        report["shards"][0]["first_line"] = 2
        (out/"catalogue.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify(out)


def test_truncated_gzip_does_not_claim_complete_coverage(tmp_path):
    source = tmp_path/"source.gz"
    source.write_bytes(gzip.compress(record()+b"\n")[:-5])
    out = tmp_path/"out"
    with pytest.raises(EOFError):
        prepare(source, out)
    report = json.loads((out/"catalogue.json").read_text())
    assert report["status"] == "failed"
    assert not report["all_input_records_retained"]


def test_no_input_cutoff_and_batch_size_invariance(tmp_path):
    source = tmp_path/"source.txt"
    source.write_bytes(b"\n".join(record() for _ in range(301)))
    a, b = prepare(source, tmp_path/"a", 7), prepare(source, tmp_path/"b", 103)
    assert a["records"] == b["records"] == 301
    assert a["source_sha256"] == b["source_sha256"]
    assert a["status_counts"] == b["status_counts"]
    assert verify(tmp_path/"a")["verified"]
    assert verify(tmp_path/"b")["verified"]


def test_catalogue_cli_does_not_enter_twelve_tracklet_reference(tmp_path, capsys):
    source = tmp_path/"source.txt"
    source.write_bytes(b"\n".join(record() for _ in range(20)))
    output = tmp_path/"out"
    assert agent_cli.main(["solve_itf", "catalogue", "prepare", str(source),
                           "--output", str(output), "--chunk-rows", "3"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["records"] == 20
    assert agent_cli.main(["solve_itf", "catalogue", "verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["verified"]
    assert agent_cli.main(["solve_itf", "catalogue", "prepare", str(source),
                           "--output", str(output)]) == 2
    assert not json.loads(capsys.readouterr().out)["ok"]
