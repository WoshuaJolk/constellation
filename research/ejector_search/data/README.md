# Retained inputs

`hunt_reffert_clusters.vot.gz` is the complete VizieR ASU response for
`J/A+A/673/A114/clusters`, retrieved 2026-10-19. It contains 7,167 rows and all
columns, including source aliases, nominal 6D kinematics, radial-velocity counts
and dispersion, age summaries and cluster radii. `cluster_request.txt` retains
the exact public URL. `provenance.json` records response metadata and hashes of
the compressed and uncompressed bytes.

The ASU service uses an INFO `matches=7167` count rather than the TAP
`QUERY_STATUS` convention. The adapter checks both the returned row count and
that metadata against the published 7,167-object catalogue, and never queries
with TOP. The fetcher refuses a changed/truncated row count. The response itself
reports a 2026-07-01 publication/update date; this is the retained version of the
catalogue associated with Hunt & Reffert (2023), not an invented 2026 study.

The stellar inputs are the existing raw Gaia/literature observations in
[`../../ejector_fit/data`](../../ejector_fit/data). Their file hashes are included
in this directory's provenance. `run.py` uses the same pre-existing quality
selection, retains all 18 eligible stars, and searches every triple. The
historical benchmark is the observed 1997 Hipparcos-era Orion triple; the three
added recovery objects are independently forward-integrated **synthetic** stars.

The generated `results/counterparts.json` retains all 5,749 usable nominal
source states and all 1,418 exclusions. Missing RVs are excluded, never set to
zero. Cluster covariance, completeness and luminosity limits are not supplied;
source matches therefore remain nominal geometric diagnostics.
