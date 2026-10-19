# Observed input provenance

`provenance.json` identifies both primary papers, table numbers and hashes of
the PDFs used for transcription. The papers are not redistributed.

`hip1997_*.json` uses Hoogerwerf et al. (2001) Table 3, J1991.25 positions,
parallax/proper-motion marginal errors and radial velocities. Correlations are
unavailable in that printed table and are explicitly approximated as zero.
`hip2007_*.json` uses the second Hipparcos reduction as printed in Bhat et al.
(2022) Table A.1, with the same fixed J1991.25 angular positions. These
approximations preclude an exact literature reproduction.

`gaia_orion.vot` and `gaia_runaways.vot` were retrieved from
`https://gea.esac.esa.int/tap-server/tap/sync` on 2026-10-19. Their exact ADQL
queries are adjacent. Requests used REQUEST=doQuery, LANG=ADQL, FORMAT=votable
and MAXREC=1000. Responses have QUERY_STATUS OK and no overflow. The three-star
query returned three matches, explicitly identified in the manifests. The
Hipparcos best-neighbour query returned 21 rows from 23 requested HIP IDs;
ι Ori is supplemented from the first query. Missing/invalid/high-RUWE inputs
are recorded in the selection ledger. Neither query applies TOP N.

`literature_rvs.json` transcribes all 23 RVs and their errors from Bhat et al.
Table A.1. Its RVs replace Gaia RVs, not averages of correlated duplicate
catalogues. This is a literature-based sample, not new spectroscopy; binary
and systemic-RV validity still requires individual checking.

`gaia_*.json` benchmark manifests use DR3 parallax/proper-motion covariance
with the angular positions held fixed, and the literature RV independently.
No Gaia parallax zero-point correction is applied in the baseline. Gaia ι Ori
has RUWE≈8.2 and is explicitly flagged as unreliable, rather than silently
treated as a precision single-star measurement.

`cluster_positions.csv` comes from the SIMBAD TAP endpoint
`https://simbad.cds.unistra.fr/simbad/sim-tap/sync`, FORMAT=csv, with the
adjacent ADQL. Cluster kinematics and printed errors in `run.py` come from
Bhat et al. Table A.2. Their errors are modeled as diagonal; angular centroid
uncertainty and cluster structure are not fitted. Using the SIMBAD J2000
centroid direction at J2016 causes a sub-milliparsec displacement at these
proper motions, much smaller than the adopted 5–7 pc passage scale, but this
is still an approximation. Point-center sensitivity results deserve additional
centroid modeling.

The campaign summary records SHA256 for every input file. Manifests and results
retain the common epoch, frame, Galactic potential, parameter bounds, and all
optimizer starts. No synthetic stars occur in this research input sample.
