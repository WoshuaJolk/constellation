"""Fetch the complete published cluster table; no credentials or stellar uploads."""
import argparse
import gzip
import io
import json
import hashlib
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode

from astropy.io.votable import parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    url = 'https://vizier.cds.unistra.fr/viz-bin/votable?'+urlencode({
        '-source': 'J/A+A/673/A114/clusters', '-out.all': '1', '-out.max': 'unlimited'})
    raw = urlopen(url, timeout=120).read()
    document = parse(io.BytesIO(raw))
    table = document.get_first_table().to_table()
    counts = [i.value for r in document.resources for i in r.infos if i.name == 'matches']
    if len(table) != 7167 or counts != ['7167']:
        raise ValueError('Expected all 7,167 published objects; refusing changed/truncated response')
    output = args.directory/'hunt_reffert_clusters.vot.gz'
    output.write_bytes(gzip.compress(raw, mtime=0))
    (args.directory/'cluster_request.txt').write_text(url+'\n')
    (args.directory/'fetch.json').write_text(json.dumps(dict(url=url, rows=len(table),
        raw_sha256=hashlib.sha256(raw).hexdigest(), compressed_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        response_metadata=[dict(name=i.name, value=i.value) for i in document.infos]), indent=2)+'\n')


if __name__ == '__main__': main()
