"""Reuse exact observation identities instead of rebuilding structured keys per fit."""
import fcntl
import json
from pathlib import Path
import numpy as np
from .catalogue import Catalogue, OBS_DTYPE
from .campaign_geometry import AtlasGeometry
from .campaign_graph import save_array, save_json
from .kernels import njit


def prepare_observation_ids(cache, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'observation-ids.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalogue = Catalogue(cache)
        identity = dict(input=catalogue.manifest, schema=1,
            key=['UTC MJD float64', 'RA mas int32', 'Dec mas int32', 'station S3'])
        marker = directory/'observation-ids.json'
        if marker.exists():
            if json.loads(marker.read_text())['identity'] != identity:
                raise ValueError('Observation-identity catalogue changed')
            return
        keys = np.empty(len(catalogue.obs), dtype=OBS_DTYPE.descr+[('site', 'S3')])
        for name in OBS_DTYPE.names:
            keys[name] = catalogue.obs[name]
        stops = np.cumsum(catalogue.tracks['count'], dtype=np.int64)
        if not np.array_equal(catalogue.tracks['start'], np.r_[0, stops[:-1]]) or stops[-1] != len(keys):
            raise ValueError('Observation blocks must partition the stored catalogue')
        keys['site'] = np.repeat(catalogue.tracks['site'], catalogue.tracks['count'])
        unique, inverse = np.unique(keys, return_inverse=True)
        save_array(directory/'observation-ids.npy', inverse.astype(np.uint32))
        save_json(marker, dict(identity=identity, observations=len(keys), unique_observations=len(unique),
            bytes=4*len(keys), complete=True))


@njit(cache=True)
def gather(indices, starts, counts, ready, identities, times, units, observers):
    size = 0
    for track in indices:
        if not ready[track]:
            raise ValueError('Observer geometry unavailable')
        size += counts[track]
    offsets = np.empty(size, np.int64); labels = np.empty(size, np.int64)
    keys = np.empty(size, np.uint32)
    cursor = 0
    for track in indices:
        for offset in range(starts[track], starts[track]+counts[track]):
            offsets[cursor] = offset; labels[cursor] = track; keys[cursor] = identities[offset]
            cursor += 1
    # Stable sort retains the first observation/label for cross-alias repeats,
    # exactly as numpy.unique(..., return_index=True) on the original keys.
    order = np.argsort(keys, kind='mergesort')
    keep = np.empty(size, np.int64); used = 0
    for j in range(size):
        if j == 0 or keys[order[j]] != keys[order[j-1]]:
            keep[used] = order[j]; used += 1
    keep = keep[:used]; offsets = offsets[keep]
    return times[offsets], units[offsets], observers[offsets], labels[keep]


class FastAtlasGeometry(AtlasGeometry):
    def __init__(self, cache, observatories, directory, identities_directory):
        super().__init__(cache, observatories, directory)
        identities_directory = Path(identities_directory)
        meta = json.loads((identities_directory/'observation-ids.json').read_text())
        if meta['identity']['input'] != self.catalogue.manifest:
            raise ValueError('Observation identities use a different catalogue')
        self.identities = np.load(identities_directory/'observation-ids.npy', mmap_mode='r')
        self.starts = np.array(self.catalogue.tracks['start'])
        self.counts = np.array(self.catalogue.tracks['count'])
        self.arrays = tuple(np.asarray(a) for a in (self.ready, self.identities, self.times, self.unit, self.observers))

    def get(self, indices):
        values = gather(np.asarray(indices, dtype=np.int64), self.starts, self.counts, *self.arrays)
        # Keep numpy's tie ordering identical to Geometry.get.
        order = np.argsort(values[0])
        return tuple(value[order] for value in values)
