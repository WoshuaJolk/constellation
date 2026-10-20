"""Reusable in-memory FITS experiment for candidate checks and injections."""

from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .io import _fits_plane, read_json, run_photometry
from .photometry import forced_photometry, sampled_psf, trailed_psf
from .search import combine_measurements


class ImageBank:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path)
        # Share strict schema, duplicate-HDU, and model validation with the CLI.
        self.original_report = run_photometry(self.path, joint=True)
        self.frames = {}
        for exposure in self.manifest["exposures"]:
            with fits.open(self.path.parent / exposure["image"], memmap=False) as hdus:
                hdu = hdus[exposure.get("image_hdu", 0)]
                data, wcs = np.array(hdu.data, dtype=float), WCS(hdu.header).celestial
            variance = exposure["variance"]
            if isinstance(variance, str):
                variance = _fits_plane(self.path.parent / variance, exposure.get("variance_hdu", 0))
            mask = _fits_plane(self.path.parent / exposure["mask"], exposure.get("mask_hdu", 0)) if "mask" in exposure else None
            kernel = _fits_plane(self.path.parent / exposure["psf_image"], exposure.get("psf_hdu", 0)) if "psf_image" in exposure else None
            self.frames[exposure["id"]] = dict(exposure=exposure, image=data, variance=variance, mask=mask, kernel=kernel, wcs=wcs)
        self.tracks = {}
        for trajectory in self.manifest["trajectories"]:
            self.tracks[trajectory["id"]] = {}
            for p in trajectory["positions"]:
                frame = self.frames[p["exposure_id"]]
                if "x" in p:
                    x, y = p["x"], p["y"]
                else:
                    x, y = frame["wcs"].world_to_pixel(SkyCoord(p["ra_deg"], p["dec_deg"], unit="deg"))
                self.tracks[trajectory["id"]][p["exposure_id"]] = dict(x=float(x), y=float(y), trail_dx=p.get("trail_dx", 0), trail_dy=p.get("trail_dy", 0))

    def sky(self, exposure_id, position):
        wcs = self.frames[exposure_id]["wcs"]
        if not wcs.has_celestial:
            raise ValueError("Sky control/injection transforms require celestial WCS in every used exposure")
        return wcs.pixel_to_world(position["x"], position["y"]).icrs

    def shift(self, track, east_arcsec, north_arcsec):
        result = {}
        for eid, p in track.items():
            coordinate = self.sky(eid, p).spherical_offsets_by(east_arcsec * u.arcsec, north_arcsec * u.arcsec)
            x, y = self.frames[eid]["wcs"].world_to_pixel(coordinate)
            result[eid] = {**p, "x": float(x), "y": float(y)}
        return result

    def template(self, eid, position, sigma_factor=1):
        frame = self.frames[eid]
        yy, xx = np.indices(frame["image"].shape)
        if frame["kernel"] is not None:
            if sigma_factor != 1:
                raise ValueError("PSF width perturbations currently require Gaussian PSFs")
            return sampled_psf(xx, yy, position["x"], position["y"], frame["kernel"], position["trail_dx"], position["trail_dy"])
        return trailed_psf(xx, yy, position["x"], position["y"], frame["exposure"]["psf_sigma_pixels"] * sigma_factor, position["trail_dx"], position["trail_dy"])

    def measure(self, track, additions=None):
        measurements, unavailable = {}, []
        for eid, position in track.items():
            frame = self.frames[eid]
            image = frame["image"]
            if additions is not None and eid in additions:
                image = image + additions[eid]
            try:
                measurement = forced_photometry(image, frame["variance"], psf_sigma=frame["exposure"]["psf_sigma_pixels"],
                                                mask=frame["mask"], psf_kernel=frame["kernel"], **position)
            except ValueError as error:
                unavailable.append({"exposure_id": eid, "reason": str(error)})
            else:
                measurements[eid] = measurement
        return measurements, unavailable

    def combine(self, measurements, trials=1, alpha=0.01):
        return combine_measurements(measurements.values(), [self.frames[eid]["exposure"].get("flux_scale", 1) for eid in measurements], trials, alpha) if measurements else None


def shared_comparison(bank, candidate, control):
    common = sorted(candidate.keys() & control.keys())
    left = bank.combine({eid: candidate[eid] for eid in common})
    right = bank.combine({eid: control[eid] for eid in common})
    return {"common_exposure_ids": common, "candidate": left, "control": right,
            "candidate_snr_minus_control_snr": left["snr"] - right["snr"] if common else None}
