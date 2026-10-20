"""Versioned discovery and JSON Schemas for the agent CLI; stdlib only."""

from copy import deepcopy

NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
STRING = {"type": "string", "minLength": 1}


def array(items, length=None, minimum=1):
    result = {"type": "array", "items": items, "minItems": minimum}
    if length is not None:
        result.update(minItems=length, maxItems=length)
    return result


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


VECTOR = array(NUMBER, 3)
MATRIX = array(array(NUMBER))
BOX = array(NUMBER, 2)
SOURCE_NAMES = ("x", "y", "flux", "vx", "vy", "parallax", "variability", "color",
                "sigma_x", "sigma_y", "rx", "ry", "rz", "vz")
EXPOSURE = obj({
    "id": STRING, "instrument": STRING, "time": NUMBER, "data": MATRIX,
    "psf_sigma": POSITIVE, "noise_sigma": {"oneOf": [POSITIVE, array(array(POSITIVE))]},
    "mask": {"oneOf": [{"type": "boolean"}, array(array({"type": "boolean"}))]},
    "affine": array(array(NUMBER, 2), 2), "offset": array(NUMBER, 2),
    "flux_scale": POSITIVE, "band_coordinate": NUMBER, "background": NUMBER,
    "parallax_x": NUMBER, "parallax_y": NUMBER, "observer": VECTOR, "center": VECTOR,
    "east": VECTOR, "north": VECTOR, "pixel_scale_arcsec": POSITIVE, "light_time_max": POSITIVE,
}, ("id", "time", "data", "psf_sigma"))
SOURCE = obj({"name": STRING, "kind": {"enum": ["star", "galaxy", "kepler"]}, "weight": POSITIVE,
              "parameters": obj({key: BOX for key in SOURCE_NAMES}, ("flux",))}, ("name", "kind", "parameters"))
SOURCE["allOf"] = [{"if": {"properties": {"kind": {"const": "kepler"}}},
                    "then": {"properties": {"parameters": {"required": ["rx", "ry", "rz", "vx", "vy", "vz"]}}},
                    "else": {"properties": {"parameters": {"required": ["x", "y"]}}}}]
COUNT = obj({"probabilities": array({"type": "number", "minimum": 0}),
             "poisson_mean": {"type": "number", "minimum": 0},
             "max_count": {"type": "integer", "minimum": 0, "maximum": 12}})
COUNT["oneOf"] = [{"required": ["probabilities"], "not": {"anyOf": [{"required": ["poisson_mean"]}, {"required": ["max_count"]}]}},
                  {"required": ["poisson_mean", "max_count"], "not": {"required": ["probabilities"]}}]
SCENE = obj({"source_types": array(SOURCE), "count_prior": COUNT, "exposures": array(EXPOSURE),
             "noise": {"enum": ["gaussian", "poisson"]},
             "nuisance": {"type": "object", "additionalProperties": BOX},
             "linear_nuisance": obj({"design": MATRIX, "mean": array(NUMBER), "covariance": MATRIX},
                                    ("design", "mean", "covariance"))},
            ("source_types", "count_prior", "exposures"))
HDU = {"oneOf": [{"type": "integer", "minimum": 0}, STRING]}
FOLD_EXPOSURE = obj({"id": STRING, "image": STRING, "variance": {"oneOf": [POSITIVE, STRING]},
                     "background": NUMBER, "image_hdu": HDU, "variance_hdu": HDU, "mask": STRING,
                     "mask_hdu": HDU, "psf_image": STRING, "psf_hdu": HDU,
                     "psf_sigma_pixels": {"type": "number", "minimum": .2, "maximum": 10},
                     "flux_scale": POSITIVE, "mjd_tdb": NUMBER, "observer_heliocentric_au": VECTOR},
                    ("id", "image", "variance", "background"))
FOLD_EXPOSURE["oneOf"] = [{"required": ["psf_sigma_pixels"], "not": {"required": ["psf_image"]}},
                          {"required": ["psf_image"], "not": {"required": ["psf_sigma_pixels"]}}]
ELEMENTS = ("a_au", "e", "inclination_deg", "node_deg", "periapsis_deg", "mean_anomaly_deg")
ORBIT = obj({"id": STRING, **{key: NUMBER for key in ELEMENTS}}, ("id", *ELEMENTS))
ORBIT["properties"].update(a_au={"type": "number", "minimum": .05, "maximum": 1000},
                            e={"type": "number", "minimum": 0, "maximum": .95},
                            inclination_deg={"type": "number", "minimum": 0, "maximum": 180})
for key in ELEMENTS[3:]:
    ORBIT["properties"][key] = {"type": "number", "minimum": -360, "maximum": 360}

SCHEMAS = {
    "fitsky": obj({"description": {"type": "string"}, "scene": SCENE, "queries": array(STRING)}, ("scene",)),
    "kepler-propagate": obj({"schema_version": {"const": 1},
        "states": array(obj({"id": STRING, "state": array(NUMBER, 6)}, ("id", "state"))),
        "times_days": array(NUMBER), "observers_heliocentric_au": array(VECTOR)},
        ("schema_version", "states", "times_days")),
    "kepler-fold": obj({"schema_version": {"const": 1}, "exposures": array(FOLD_EXPOSURE),
        "orbits": array(ORBIT), "epoch_mjd_tdb": NUMBER,
        "trajectories": array(obj({"id": STRING, "xy": array(array(NUMBER, 2))}, ("id", "xy"))),
        "cell_subdivisions": {"enum": [1, 2, 4, 8, 16]}, "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 1000}, "validate_direct": {"type": "boolean"},
        "benchmark_repeats": {"type": "integer", "minimum": 0, "maximum": 9}}, ("schema_version", "exposures")),
    "kepler-chart": obj({"schema_version": {"const": 1}, "arrays": STRING,
        "chart": obj({**{k: VECTOR for k in ["observer", "velocity", "normal", "east", "north"]},
                      "retarded_epoch": {"type": "boolean"}}, ("observer", "velocity", "normal", "east", "north")),
        "center": array(NUMBER, 6), "widths": array(POSITIVE, 6), "position_tolerance_arcsec": POSITIVE,
        "phase_tolerance": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "max_leaves": {"type": "integer", "minimum": 1, "maximum": 4096}, "max_seconds": POSITIVE,
        "outer_degree": {"enum": [2, 4]}, "translated_certificate": {"type": "boolean"},
        "allow_direct_fallback": {"type": "boolean"}, "audit_count": {"type": "integer", "minimum": 0, "maximum": 32},
        "route": {"enum": ["auto", "polynomial", "slices"]}},
        ("schema_version", "arrays", "chart", "center", "widths", "position_tolerance_arcsec")),
}
SCHEMAS["kepler-fold"]["oneOf"] = [
    {"required": ["orbits", "epoch_mjd_tdb"], "not": {"required": ["trajectories"]},
     "properties": {"exposures": {"items": {"required": ["mjd_tdb", "observer_heliocentric_au"]}}}},
    {"required": ["trajectories"], "not": {"anyOf": [{"required": ["orbits"]}, {"required": ["epoch_mjd_tdb"]}]}},
]

TOOLS = {
    "fitsky": dict(summary="Joint sequential pixel inference with unknown sources, orbits, blends and shared calibration",
                   dependencies=["numpy", "scipy", "astropy", "flint"], accuracy="approximate_uncertified",
                   limits="Can miss posterior modes; two-body orbits, instantaneous Gaussian PSFs, supplied calibration model"),
    "kepler-propagate": dict(summary="Propagate Cartesian states; optionally solve observer-dependent light time",
                   dependencies=["numpy", "scipy", "astropy"], accuracy="numerical_two_body",
                   limits="No perturbations, image response, trajectory search or posterior inference"),
    "kepler-fold": dict(summary="Search a supplied finite orbit/trajectory bank against FITS pixels",
                   dependencies=["numpy", "scipy", "astropy"], accuracy="finite_bank_only",
                   limits="Elliptic elements or supplied pixel tracks; fixed response/background; not the PFKT time-compression research engine"),
    "kepler-chart": dict(summary="Apply the bounded-domain chart research transform to supplied Fourier fields",
                   dependencies=["numpy", "scipy", "astropy", "flint", "finufft"], accuracy="research_conditional",
                   limits="Fourier-field adapter, not raw-pixel ingestion or archive-wide FKT; full score error is not certified"),
}

SCHEMAS["kepler-propagate"]["properties"]["states"]["description"] = (
    "Rows contain unique id and [rx,ry,rz,vx,vy,vz] at a common zero epoch; heliocentric ICRS-equatorial AU and AU/day")
SCHEMAS["kepler-propagate"]["properties"]["times_days"]["description"] = "TDB day offsets from the input-state epoch"
SCHEMAS["kepler-propagate"]["properties"]["observers_heliocentric_au"]["description"] = (
    "Optional reception observer positions in the same frame, one per time; supplying these enables light time and RA/Dec")
SCHEMAS["kepler-chart"]["properties"]["arrays"]["description"] = (
    "NPZ relative to manifest: times(T,), observers(T,3), visits(M,) integer exposure indices, "
    "frequencies(M,2) radians^-1, data(M,) complex Fourier coefficients, parameters(N,6); "
    "optional normal/east/north each (T,3). Observer-chart parameter order: inverse axial distance, "
    "fractional axial speed, north angular rate, east tangent coordinate, north tangent coordinate, east angular rate")
EXPOSURE["properties"]["time"] = dict(NUMBER, description="TDB days from source-state zero epoch; not absolute MJD")
EXPOSURE["properties"]["observer"] = dict(VECTOR, description="Heliocentric reception position in AU, in the source-state inertial frame")


def schema(tool):
    result = deepcopy(SCHEMAS[tool])
    result.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": tool,
                   "description": TOOLS[tool]["limits"]})
    return result
