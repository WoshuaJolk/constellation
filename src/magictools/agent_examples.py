"""Small runnable synthetic inputs, generated into a newly owned directory."""


def create(tool, directory):
    if tool == "solve_itf":
        from .solve_itf_example import example
        return example()
    if tool == "fit_sky":
        from research.fit_sky.orbital_continuous_control import scene, QUERIES
        return {"description": "Synthetic 24-day ambiguity control, not real-survey data", "scene": scene(), "queries": QUERIES}
    if tool == "kepler-propagate":
        return {"schema_version": 1, "states": [{"id": "example", "state": [3., 0., 0., 0., .01, 0.]}],
                "times_days": [0., 1., 8.], "observers_heliocentric_au": [[1., 0., 0.]]*3}
    import numpy as np
    if tool == "kepler-chart":
        center = np.array([.5, 0., 0., 0., 0., -.001])
        widths = np.full(6, 1e-8)
        np.savez_compressed(directory/"fields.npz", times=np.array([-.02, .02]),
                            observers=np.array([[1., 0., 0.]]*2), visits=np.array([0, 0, 1, 1]),
                            frequencies=np.array([[1e4, 0.], [0., 1e4]]*2), data=np.array([1., .5, 1., .5], complex),
                            parameters=np.stack([center, center+widths*.25]))
        return {"schema_version": 1, "arrays": "fields.npz", "center": center.tolist(), "widths": widths.tolist(),
                "chart": {"observer": [1., 0., 0.], "velocity": [0., .0172, 0.],
                          "normal": [1., 0., 0.], "east": [0., 1., 0.], "north": [0., 0., 1.]},
                "position_tolerance_arcsec": .001, "max_leaves": 1, "max_seconds": 1.,
                "allow_direct_fallback": False, "audit_count": 1}
    from astropy.io import fits
    from astropy.wcs import WCS
    from .fold_kepler import project_kepler, ELEMENTS
    from .photometry import gaussian_psf
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval, wcs.wcs.crpix, wcs.wcs.cdelt = [0., 0.], [11., 11.], [-30/3600, 30/3600]
    elements = np.array([[3., 0., 0., 0., 0., angle] for angle in [-.001, 0., .001]])
    times = np.array([60000., 60000.05])
    angle = .0172*(times-times[0])
    observers = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(2)])
    positions = project_kepler(elements, times[0], times, observers, [wcs]*2)
    yy, xx = np.indices((21, 21))
    exposures = []
    for i in range(2):
        image = 20*gaussian_psf(xx, yy, *positions[1, i], 1.)
        fits.PrimaryHDU(image, header=wcs.to_header()).writeto(directory/f"frame-{i}.fits")
        exposures.append({"id": f"e{i}", "image": f"frame-{i}.fits", "variance": 1., "background": 0.,
                          "psf_sigma_pixels": 1., "mjd_tdb": float(times[i]), "observer_heliocentric_au": observers[i].tolist()})
    return {"schema_version": 1, "epoch_mjd_tdb": float(times[0]), "exposures": exposures,
            "orbits": [{"id": f"orbit-{i}", **dict(zip(ELEMENTS, row.tolist()))} for i, row in enumerate(elements)],
            "validate_direct": True, "top_k": 3}
