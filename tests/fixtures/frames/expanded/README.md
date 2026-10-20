# Expanded archive protocol fixtures

Public metadata retrieved on 2026-10-20. These are deliberately small subsets,
not completeness claims. VOTable TABLEDATA rows were trimmed while retaining
native column metadata. Herschel's compact binary table is retained intact.
Gzip files use a fixed timestamp for reproducibility.

Sources and validation query results are documented in `docs/frames.md`.
The filenames identify each service. `noirlab-*.json` covers the archive's
instrument aliases. `akari.html` includes a genuine row with no product link;
`dasch.json` is the native CSV-lines JSON format. `scanpi-*` preserves the XML
index and four band tables; their temporary product links may expire. The
MAST fixtures contain observation metadata and a target-pixel product list.

LCO's temporary S3 signed URLs were replaced with example.invalid and its
version_set removed. Tests use the stable native frame metadata endpoint.
`igaps.fits` contains three real metadata rows (no image pixels) from the
527,736-row public HTTP index, including UTC utstart timestamps. No credentials
or image pixels are included. HSC and Rubin have no invented authenticated
response fixtures: account limitations remain explicit.
