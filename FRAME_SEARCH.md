# Constellation frame finder

The interface is at `/constellation/search`, linked from the constellation
inventory. The existing Vercel deployment and host rewrite remain in use.
No database, credentials, image downloads, or new hosting service are required.

## Inputs and adapters

| Input | Connected services | Matching behavior |
| --- | --- | --- |
| Sky field | CADC Argus ObsCore, ESO ObsCore | ICRS circular field intersects indexed footprint; product time interval overlaps requested dates |
| Sky field | IRSA ZTF IBE | Bounding square of side twice the radius intersects science-image footprint |
| Known Solar System object | CADC SSOIS, IRSA MOST (ZTF) | Archive service resolves designation and computes predicted positions |
| Candidate elliptic orbits | CADC SSOIS, IRSA MOST (ZTF) | Each supplied heliocentric J2000 orbit is queried separately |

The field input is target-agnostic. Known-object name resolution is specifically
for Solar System objects; use coordinates for fixed objects. Hyperbolic orbits,
uploaded ephemerides, polygon fields, and arbitrarily large batch queries are not
implemented. The dataset inventory is broader than the connected services and
must not be interpreted as a list of datasets searched.

The initial request limits are a 0.001–2 degree radius, 3,660 days, 1–3 orbit
hypotheses, and 300 products per provider/hypothesis. A limit produces an explicit
`limited` receipt; split the field/date interval to retrieve more. The selection
is not a detectability ranking or a guaranteed chronological subset. Returned
rows are sorted chronologically after retrieval. Overlapping archives may assign
different identifiers to the same image, so cross-archive duplicates can remain.

Dates use UTC start-inclusive/end-exclusive boundaries for field queries and
MOST results. SSOIS receives calendar boundaries and retains provider-defined
timestamp conventions, including TAI for some products. Do not use SSOIS MJD
as UTC without consulting the native archive. Orbital epochs are MJD passed
unchanged to the orbit services; there is no time-scale or reference-frame
conversion. Supply elements in their expected convention. No local orbit
propagator or uncertainty model is used.

Matches are candidate image products, not detections. Archive footprints may
include detector gaps. Masks, depth, trailing, and photometric suitability are
not assessed. ObsCore rows may represent stacks or mosaics with broad time
intervals. File availability/rights are not independently verified. FITS links
are direct where available; DataLink links return a product-link table.

## Agent API

`GET /api/frames` returns capabilities, limits, conventions, and an example.
`POST /api/frames` accepts JSON:

```json
{
  "mode": "field",
  "ra": 180,
  "dec": 30,
  "radius": 0.05,
  "start": "2019-01-28",
  "end": "2019-01-29",
  "providers": ["cadc", "eso", "ztf"]
}
```

For a known object, use `"mode": "object"`, `"object": "2006 WG206"`,
dates `2007-01-18` to `2007-01-19`, and providers `["ssois", "most"]`.
Omit field coordinates. For hypotheses:

```json
{
  "mode": "orbits",
  "start": "2019-01-01",
  "end": "2019-02-01",
  "providers": ["ssois", "most"],
  "orbits": [{
    "label": "candidate-1",
    "epochMjd": 55200,
    "a": 2.7413806,
    "e": 0.0855139,
    "i": 6.98707,
    "node": 43.05408,
    "peri": 120.67443,
    "meanAnomaly": 206.88213
  }]
}
```

`a` is in au; angles are in degrees; `0 <= e < 1`. `providers` defaults
to all adapters supported by the mode. Labels must be unique.

The JSON response includes normalized `frames`, original row metadata (`raw`),
the submitted `input`, a UTC `queriedAt`, per-query `checks` with exact query URLs
and `ok`/`limited`/`error` status, adapters `unqueried`, and `limitations`.
Repeated frames across hypotheses are merged and retain their hypothesis labels.
Hypothesis counts are unweighted sample counts, not probabilities.

Invalid input returns HTTP 400. Archive failures return HTTP 200 with an `error`
check so successful providers can still return results. Consumers **must inspect
every check**. Neither an error nor a successful empty query proves no
observations exist elsewhere. JSON export preserves the entire receipt;
CSV exports frame rows with per-frame query status and should be accompanied by
the JSON receipt when sharing results.

## Operation and validation

Run `npm ci`, `npm test`, and `npm run build`. `npm run dev` starts the interface.
The API uses fixed archive endpoints, 22-second fetch deadlines, three concurrent
jobs, a 6 MB response cap, a 16 KB request cap, and a 60-second Next route budget.
There is no persistent cache. If deploying at high request volume, add
deployment-level rate limiting and a shared query cache before broad automation.

Tests cover real archived response fixtures, table corruption, date boundaries,
validation, retrieval URLs, partial outages, row limits, orbit deduplication,
and safe exports. Fixtures were retrieved on 2026-10-20; tests do not require
network access. Live smoke tests should be treated as observations of service
availability, not a guarantee that an external archive is always available.

Protocol references:

- [CADC TAP / Argus](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/doc/tap/)
- [ESO ObsCore examples](https://archive.eso.org/tap_obs/examples)
- [ZTF IBE](https://irsa.ipac.caltech.edu/ibe/docs/ztf.html)
- [SSOIS documentation](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/documentation.html)
- [MOST program interface](https://irsa.ipac.caltech.edu/applications/MOST/MOSTProgramInterface.html)
