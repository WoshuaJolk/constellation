# Archive protocol fixtures

These are public metadata responses, not image pixels or credentials. Except
for S60, the data are small subsets for deterministic offline tests, not
completeness claims.

- `s60.tsv.gz`: full S60 pinned-object SSOIS response from the existing web
  repo's `moonshots/audit/ssois/S60.tsv`, copied 2026-10-20. 12,218 data rows,
  12,219 lines including header. No first-N truncation.
- `ssois.tsv`, `ztf.tbl`, `most-orbit.tbl`, `cadc.csv`: public metadata fixtures
  retained from the constellation frame-search prototype. The ZTF fixture has
  six science-quadrant records for 2019-01-28.
- `noirlab.json`: three real DECam instcal records from NOIRLab `adv_search`,
  queried 2026-10-20 around RA 150, Dec 2. Query metadata records a `caldat`
  selection; the adapter uses `dateobs_center` to avoid confusing an observing
  night with UTC exposure time.
- `smoka.txt`: ASCII header and first four real HSC CCD records from a
  2015-01-16 field search, retrieved 2026-10-20. Extracted from the frame-table
  `<pre>` block; HTML and unrelated search-summary blocks omitted.
- `mast.json`: first two records of a real 133-record RA 150, Dec 2 query for
  2010–2020, retrieved 2026-10-20. Paging metadata adjusted for this subset.
- `catch.json`: first two records of a real 770-product RA 180, Dec 0 query for
  2020–2021, retrieved 2026-10-20. `count` adjusted to the subset.
- `wise.tbl`: full 12-row MOST `wise_merge` result for Victoria,
  2014-05-01–2014-05-30, retrieved 2026-10-20. Different bands must stay distinct.
- `hsc-schema.json`: public PDR3 wide `frame` schema, retrieved 2026-10-20 from
  https://hsc-release.mtk.nao.ac.jp/schema/schemata/pdr3.pdr3_wide.frame.json.
  No authenticated HSC results are claimed.

Tests generate malformed rows, overflow responses and additional paging rows
from these fixtures to exercise failure behavior without querying the archives.
