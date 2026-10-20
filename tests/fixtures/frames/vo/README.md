# Native VO and KASI metadata fixtures

Recorded from public archive services on 2026-10-20. These contain metadata,
not image pixels or credentials. Gzip files decompress to the original text
response; the `binary-arrays` VOTable contains an embedded binary metadata table.
Counts below describe these bounded queries, not archive volumes/completeness.

| Fixture | Source query / scope |
| --- | --- |
| alma | `https://almascience.eso.org/tap/sync`, ObsCore cone (286.4842, -0.0352), radius .001 deg, 2023-05-20 to 2023-05-22; 10 science products |
| casda | `https://casda.csiro.au/casda_vo_tools/tap/sync`, ObsCore cone (150.8377, -29.2867), .001 deg, 2020; 134 records including undated products |
| gavo | `https://dc.g-vo.org/tap/sync`, cone (11.61996, 42.19246), .001 deg, September 2011; 49 native rows before local filtering |
| chandra | `https://cda.harvard.edu/cxcsiap/queryImages`, SIA1 (83.633, 22.0145), .01 deg circular search converted to a bounding rectangle; all dates returned, locally filtered |
| xmm, xmm-slew | `https://nxsa.esac.esa.int/nxsa-sl/servlet/siap` and `siap-slew`, same position, .001 deg; native internal-host URLs are intentionally retained to test public URL repair |
| heasarc | `https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync`, `SELECT TOP 3 * FROM ivoa.obscore WHERE dataproduct_type='image'`; native ADQL geometry strings |
| nrao | `https://data-query.nrao.edu/tap/sync`, TOP 3 image rows in a 5-degree cone around (180,30), `t_min>59000 AND t_max<59001`; demonstrates the archive's invalid zero end times |
| kmtnet | `https://data.kasi.re.kr/api/KMTNet/search`, (270.694,-30.066), 1.5 deg, limit=3; bounded metadata sample only |
| miris | `https://data.kasi.re.kr/api/MIRIS/search`, (84,-70), 3.01 deg, limit=2000000, datatype=OBJECT; 23 metadata rows |
| binary-arrays | XSA TAP `SELECT TOP 3 * FROM xsa.v_public_observations`; regression for array-valued masks in binary VOTables |
| registry | RegTAP at GAVO, SIA/TAP resources with Chandra/HEASARC/XMM/SkyMapper in their title |
| inspect | ALMA TAP_SCHEMA.columns for ivoa.obscore; 73 columns |
| links | ALMA DataLink for a member OUS from the ALMA sample; seven links/service entries |

Production searches do not use TOP 3. The KASI adapter requests its complete
bounded cone up to the explicit server request boundary, and reports exhaustion
as unverified because that API supplies no independent total/overflow marker.
