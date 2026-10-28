"""Compact, memory-mapped ITF ingestion with explicit accounting of every row."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .kernels import parse_numeric, summaries

OBS_DTYPE = np.dtype([('t', '<f8'), ('ra_mas', '<i4'), ('dec_mas', '<i4')])
TRACK_DTYPE = np.dtype([('id', 'S12'), ('site', 'S3'), ('start', '<i8'), ('count', '<i4'),
                       ('t', '<f8'), ('u', '<f8', 3), ('v', '<f8', 3),
                       ('rms', '<f4'), ('span', '<f4')])
SCHEMA = 2


def unit_vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(source, destination, progress=print):
    """Retain all supported positions; no magnitude, rate or RMS selection.

    MPC identifiers may be reused over nights: split at gaps > 0.5 day.
    Position quantization is at most 0.5 mas per coordinate; times stay float64.
    Original source is untouched. Destination is atomically published once complete.
    """
    start = time.monotonic()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    fingerprint = sha256(source)
    if destination.exists():
        manifest = json.loads((destination/'manifest.json').read_text())
        if manifest['schema'] != SCHEMA or manifest['source_sha256'] != fingerprint:
            raise ValueError('Cache belongs to different input/schema; choose a new --cache directory')
        return Catalogue(destination)
    counts = Counter()
    arrays, identifiers, sites = [], [], []
    opener = gzip.open if source.suffix == '.gz' else open
    with opener(source, 'rb') as stream:
        while True:
            lines = stream.readlines(8*1024*1024)
            if not lines:
                break
            counts['source_rows'] += len(lines)
            fixed = [line.rstrip(b'\r\n') for line in lines]
            good = [line for line in fixed if len(line) == 80]
            counts['invalid_width_rows'] += len(fixed)-len(good)
            if not good:
                continue
            raw = np.frombuffer(b''.join(good), np.uint8).reshape(-1, 80)
            values, status = parse_numeric(raw)
            counts['invalid_numeric_rows'] += int(np.sum(status == 1))
            counts['unsupported_observer_rows'] += int(np.sum(status == 2))
            valid = status == 0
            ids = np.ndarray((len(raw),), dtype='S12', buffer=raw, strides=(80,)).copy()[valid]
            site = np.ndarray((len(raw),), dtype='S3', buffer=raw, offset=77, strides=(80,)).copy()[valid]
            blank = np.char.strip(ids) == b''
            counts['anonymous_rows'] += int(blank.sum())
            # Unnamed observations cannot be asserted to form one tracklet.
            if np.any(blank):
                for k in np.flatnonzero(blank):
                    ids[k] = f'?{counts["accepted_rows"]+k:011d}'.encode()
            arrays.append(values[valid])
            identifiers.append(ids)
            sites.append(site)
            counts['accepted_rows'] += int(valid.sum())
    if not arrays:
        raise ValueError('No supported optical observations in input')
    values = np.concatenate(arrays)
    ids, site = np.concatenate(identifiers), np.concatenate(sites)
    del arrays, identifiers, sites
    progress(f'Parsed {counts["source_rows"]:,} rows; indexing {len(values):,} optical observations')
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0], site, ids))
    values, ids, site = values[order], ids[order], site[order]
    packed = np.empty(len(values), OBS_DTYPE)
    packed['t'] = values[:, 0]
    packed['ra_mas'] = np.rint(values[:, 1]*3600000).astype(np.int32) % 1296000000
    packed['dec_mas'] = np.rint(values[:, 2]*3600000).astype(np.int32)
    del values, order
    same_key = (ids[1:] == ids[:-1]) & (site[1:] == site[:-1])
    # Exact duplicate measurements within a key must not buy extra significance.
    duplicate = np.r_[False, same_key & (packed[1:] == packed[:-1])]
    counts['duplicate_rows_within_key'] = int(duplicate.sum())
    packed, ids, site = packed[~duplicate], ids[~duplicate], site[~duplicate]
    new = np.r_[True, (ids[1:] != ids[:-1]) | (site[1:] != site[:-1]) |
                (np.diff(packed['t']) > .5)]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(packed)]
    u = unit_vectors(packed['ra_mas']/3600000, packed['dec_mas']/3600000)
    stats = summaries(packed['t'], u, starts, ends)
    tracks = np.empty(len(starts), TRACK_DTYPE)
    tracks['id'], tracks['site'] = ids[starts], site[starts]
    tracks['start'], tracks['count'] = starts, ends-starts
    tracks['t'], tracks['u'], tracks['v'] = stats[:, 0], stats[:, 1:4], stats[:, 4:7]
    tracks['rms'], tracks['span'] = stats[:, 7], stats[:, 8]
    counts['stored_observations'] = len(packed)
    counts['tracklets'] = len(tracks)
    counts['singletons'] = int(np.sum(tracks['count'] == 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name+'.tmp-', dir=destination.parent))
    try:
        np.save(staging/'observations.npy', packed, allow_pickle=False)
        np.save(staging/'tracklets.npy', tracks, allow_pickle=False)
        # Portable compressed representation as well as mmap working files.
        write_archive(staging/'catalogue.npz', packed, tracks)
        manifest = dict(schema=SCHEMA, source=str(source), source_sha256=fingerprint,
                        source_bytes=source.stat().st_size, counts=dict(counts),
                        time_scale='MJD UTC', angle_unit='integer milliarcsecond',
                        maximum_coordinate_quantization_arcsec=.0005,
                        archive_time_unit='integer microday; native MPC six-place day precision',
                        working_bytes=sum((staging/name).stat().st_size for name in
                                          ['observations.npy', 'tracklets.npy']),
                        archive_bytes=(staging/'catalogue.npz').stat().st_size,
                        archive_sha256=sha256(staging/'catalogue.npz'),
                        build_seconds=time.monotonic()-start)
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    progress(f'Cached {len(tracks):,} tracklets in {manifest["build_seconds"]:.1f}s')
    return Catalogue(destination)


def write_archive(path, observations, tracks):
    """Delta-code native-precision observations; recompute redundant summaries on restore."""
    np.savez_compressed(path,
        time_delta_microday=np.diff(np.rint(observations['t']*1e6).astype(np.int64), prepend=np.int64(0)),
        ra_delta_mas=np.diff(observations['ra_mas'], prepend=np.int32(0)),
        dec_delta_mas=np.diff(observations['dec_mas'], prepend=np.int32(0)),
        ids=tracks['id'], sites=tracks['site'], counts=tracks['count'])


def restore(archive, manifest_path, destination):
    """Expand the portable NPZ + manifest into mmap arrays without the original MPC dump."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Restore destination already exists')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported ITF cache schema')
    if manifest.get('archive_sha256') != sha256(archive):
        raise ValueError('Compact archive checksum does not match manifest')
    with np.load(archive, allow_pickle=False) as data:
        counts = data['counts']
        nobs, ntracks = manifest['counts']['stored_observations'], manifest['counts']['tracklets']
        if (counts.shape != (ntracks,) or counts.dtype.kind not in 'iu' or
                np.any(counts < 1) or int(np.sum(counts, dtype=np.int64)) != nobs or
                data['ids'].shape != (ntracks,) or data['ids'].dtype != np.dtype('S12') or
                data['sites'].shape != (ntracks,) or data['sites'].dtype != np.dtype('S3')):
            raise ValueError('Invalid compact archive tracklet arrays')
        for field in ('time_delta_microday', 'ra_delta_mas', 'dec_delta_mas'):
            if data[field].shape != (nobs,) or data[field].dtype.kind != 'i':
                raise ValueError('Invalid compact archive observation arrays')
        packed = np.empty(nobs, OBS_DTYPE)
        packed['t'] = np.cumsum(data['time_delta_microday'], dtype=np.int64)/1e6
        ra = np.cumsum(data['ra_delta_mas'], dtype=np.int64)
        dec = np.cumsum(data['dec_delta_mas'], dtype=np.int64)
        if np.any((ra < 0) | (ra >= 1296000000) | (abs(dec) > 324000000)):
            raise ValueError('Invalid compact archive angles')
        packed['ra_mas'], packed['dec_mas'] = ra, dec
        ends = np.cumsum(counts, dtype=np.int64)
        starts = np.r_[0, ends[:-1]]
        u = unit_vectors(ra/3600000, dec/3600000)
        stats = summaries(packed['t'], u, starts, ends)
        tracks = np.empty(ntracks, TRACK_DTYPE)
        tracks['id'], tracks['site'], tracks['count'] = data['ids'], data['sites'], counts
        tracks['start'], tracks['t'] = starts, stats[:, 0]
        tracks['u'], tracks['v'] = stats[:, 1:4], stats[:, 4:7]
        tracks['rms'], tracks['span'] = stats[:, 7], stats[:, 8]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name+'.tmp-', dir=destination.parent))
    try:
        np.save(staging/'observations.npy', packed, allow_pickle=False)
        np.save(staging/'tracklets.npy', tracks, allow_pickle=False)
        shutil.copyfile(archive, staging/'catalogue.npz')
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return Catalogue(destination)


class Catalogue:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if self.manifest['schema'] != SCHEMA:
            raise ValueError('Unsupported ITF cache schema')
        self.obs = np.load(self.path/'observations.npy', mmap_mode='r', allow_pickle=False)
        self.tracks = np.load(self.path/'tracklets.npy', mmap_mode='r', allow_pickle=False)
        if (self.obs.dtype != OBS_DTYPE or self.tracks.dtype != TRACK_DTYPE or
                len(self.obs) != self.manifest['counts']['stored_observations'] or
                len(self.tracks) != self.manifest['counts']['tracklets']):
            raise ValueError('Cache shape or type does not match manifest')

    def observations(self, indices):
        blocks, sites, labels = [], [], []
        for index in indices:
            track = self.tracks[index]
            blocks.append(self.obs[track['start']:track['start']+track['count']])
            sites.extend([track['site']]*track['count'])
            labels.extend([index]*track['count'])
        data, sites, labels = np.concatenate(blocks), np.array(sites), np.array(labels)
        # Cross-key resubmissions of the same station/time/position count once.
        key = np.empty(len(data), dtype=OBS_DTYPE.descr+[('site', 'S3')])
        for name in OBS_DTYPE.names:
            key[name] = data[name]
        key['site'] = sites
        _, unique = np.unique(key, return_index=True)
        unique = unique[np.argsort(data['t'][unique])]
        return data[unique], sites[unique], labels[unique]
