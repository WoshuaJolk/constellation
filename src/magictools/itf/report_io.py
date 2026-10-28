"""Streaming access to candidate reports; never materialize the candidate array."""
from pathlib import Path
import ijson


def metadata(path):
    builder = ijson.ObjectBuilder()
    with Path(path).open('rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == 'candidates' or prefix.startswith('candidates.'):
                continue
            if prefix == '' and event == 'map_key' and value == 'candidates':
                continue
            builder.event(event, value)
    return builder.value


def candidates(path):
    with Path(path).open('rb') as stream:
        yield from ijson.items(stream, 'candidates.item', use_float=True)
