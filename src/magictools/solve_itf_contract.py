"""Strict input contract for the solve_itf research reference; stdlib only."""

def obj(properties, required):
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}

def arr(items, minimum=1, maximum=None):
    value = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        value["maxItems"] = maximum
    return value

NUMBER = {"type": "number"}
ID = {"type": "string", "minLength": 1, "maxLength": 200}
VECTOR = arr(NUMBER, 3, 3)
OBSERVATION = obj({
    "id": ID, "observatory": ID, "mjd_tdb": NUMBER,
    "ra_deg": {"type": "number", "minimum": 0, "exclusiveMaximum": 360},
    "dec_deg": {"type": "number", "minimum": -90, "maximum": 90},
    "sigma_arcsec": {"type": "number", "minimum": .001, "maximum": 648000},
    "observer_heliocentric_au": VECTOR,
}, ["id", "observatory", "mjd_tdb", "ra_deg", "dec_deg", "sigma_arcsec",
    "observer_heliocentric_au"])
SCHEMA = obj({
    "schema_version": {"const": 1},
    "snapshot_id": ID,
    "epoch_mjd_tdb": NUMBER,
    "frame": {"const": "heliocentric_icrs"},
    "tracklets": arr(obj({"id": ID, "observations": arr(OBSERVATION, 2, 32)},
                         ["id", "observations"]), 1, 12),
    "orbit_prior": arr(obj({
        "id": ID, "state": arr(NUMBER, 6, 6),
        "weight": {"type": "number", "exclusiveMinimum": 0},
    }, ["id", "state", "weight"]), 1, 4096),
    "link_activity": {"type": "number", "minimum": 1e-100, "maximum": 1e100},
    "queries": arr(arr(ID, 2, 12), 0, 100),
    "max_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 180},
    "max_nodes": {"type": "integer", "minimum": 0, "maximum": 100000},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
    "target_unresolved_mass": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
}, ["schema_version", "snapshot_id", "epoch_mjd_tdb", "frame", "tracklets",
    "orbit_prior", "link_activity"])
