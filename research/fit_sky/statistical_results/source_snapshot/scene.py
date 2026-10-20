"""Explicit multi-epoch image likelihood and exchangeable catalogue prior.

The implemented instruments have pixel-integrated axis-aligned Gaussian PSFs,
affine tangent-plane geometry and Gaussian or Poisson measurements. This is an
extensible research model, not a model of every instrument.
"""

from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from flint import arb

from .arithmetic import Jet, ball, square


def _constant(x):
    return x.c[0] if isinstance(x, Jet) else x


def _finite(value):
    return np.isfinite(value) if isinstance(value, (int, float)) else False


def _cholesky(c):
    n = len(c)
    result = [[arb(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            v = c[i][j]-sum((result[i][k]*result[j][k] for k in range(j)), arb(0))
            if i == j:
                if not v > 0:
                    raise ValueError("Covariance not certified positive definite")
                result[i][j] = v.sqrt()
            else:
                result[i][j] = v/result[j][j]
    return result


class Scene:
    """All numeric manifest inputs are interpreted as exact binary floats."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.types = manifest["source_types"]
        self._fixed_orbit_cache = {}
        self.exposures = manifest["exposures"]
        if not self.types or not self.exposures:
            raise ValueError("Source classes and exposures required")
        self.nuisance = manifest.get("nuisance", {})
        self.noise = manifest.get("noise", "gaussian")
        if self.noise not in ("gaussian", "poisson"):
            raise ValueError("Only Gaussian and Poisson experiments are implemented")
        allowed = {"star", "galaxy", "kepler"}
        names = [t["name"] for t in self.types]
        if len(set(names)) != len(names) or any(not isinstance(n, str) or ":" in n for n in names):
            raise ValueError("Unique source-class names without colons required")
        for source in self.types:
            if source["kind"] not in allowed:
                raise ValueError("Unknown source class")
            params = source["parameters"]
            required = {"flux"} | ({"rx", "ry", "rz", "vx", "vy", "vz"}
                                    if source["kind"] == "kepler" else {"x", "y"})
            if not required <= params.keys():
                raise ValueError(f"Missing source parameters: {required-params.keys()}")
            allowed_parameters = ({"rx", "ry", "rz", "vx", "vy", "vz", "flux", "color", "variability"}
                                  if source["kind"] == "kepler" else
                                  {"x", "y", "vx", "vy", "parallax", "flux", "color", "variability",
                                   "sigma_x", "sigma_y"} if source["kind"] == "galaxy" else
                                  {"x", "y", "vx", "vy", "parallax", "flux", "color", "variability"})
            if not params.keys() <= allowed_parameters:
                raise ValueError("Unsupported source parameter")
            if params["flux"][0] < 0:
                raise ValueError("Flux prior must be nonnegative")
            for key in ("sigma_x", "sigma_y"):
                if key in params and params[key][0] < 0:
                    raise ValueError("Intrinsic sizes must be nonnegative")
        for intervals in [self.nuisance]+[t["parameters"] for t in self.types]:
            for name, interval in intervals.items():
                if (not isinstance(interval, list) or len(interval) != 2
                        or not all(_finite(x) for x in interval) or interval[0] > interval[1]):
                    raise ValueError(f"Finite ordered bounds required: {name}")
        ids = [e["id"] for e in self.exposures]
        if len(set(ids)) != len(ids):
            raise ValueError("An exposure must enter only once")
        cameras = {e.get("instrument", "default") for e in self.exposures}
        supported_nuisance = {c+"_"+p for c in cameras
                              for p in ["dx", "dy", "log_gain", "log_psf", "background"]}
        if not self.nuisance.keys() <= supported_nuisance:
            raise ValueError("Unknown or unused instrument nuisance parameter")
        self.data, self.sigmas, self.active = [], [], []
        for exposure in self.exposures:
            data = np.asarray(exposure["data"], float)
            if data.ndim != 2 or not np.all(np.isfinite(data)):
                raise ValueError("Finite two-dimensional pixel data required")
            mask = np.broadcast_to(np.asarray(exposure.get("mask", False), bool), data.shape)
            sigma = np.broadcast_to(np.asarray(exposure.get("noise_sigma", 1.), float), data.shape)
            psf = exposure["psf_sigma"]
            if (not _finite(psf) or psf <= 0 or not _finite(exposure["time"])
                    or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
                raise ValueError("Positive PSF/noise and finite time required")
            if self.noise == "poisson" and (np.any(data < 0) or np.any(data != np.floor(data))):
                raise ValueError("Poisson data must be nonnegative integer counts")
            self.active.append([(j, i) for j, i in np.ndindex(data.shape) if not mask[j, i]])
            self.data.extend(float(data[j, i]) for j, i in self.active[-1])
            self.sigmas.extend(float(sigma[j, i]) for j, i in self.active[-1])
        if not self.data:
            raise ValueError("At least one usable pixel required")
        self.cholesky = None
        self.low_rank = None
        self.log_normalization = arb(0)
        if self.noise == "gaussian":
            self.log_normalization = -len(self.data)/2*arb(2*arb.pi()).log()
            covariance = manifest.get("covariance")
            linear = manifest.get("linear_nuisance")
            if covariance is None and linear is not None:
                design = np.asarray(linear["design"], float)
                mean = np.asarray(linear["mean"], float)
                cov = np.asarray(linear["covariance"], float)
                n, k = len(self.data), len(mean)
                if (mean.ndim != 1 or k == 0 or design.shape != (n, k)
                        or cov.shape != (k, k) or not np.array_equal(cov, cov.T)
                        or not all(np.all(np.isfinite(x)) for x in [design, mean, cov])):
                    raise ValueError("Valid shared Gaussian linear nuisance required")
                prior_chol = _cholesky([[arb(float(x)) for x in row] for row in cov])
                self.data = [arb(y)-sum((arb(float(design[i, j]))*arb(float(mean[j]))
                                       for j in range(k)), arb(0)) for i, y in enumerate(self.data)]
                t = [[sum((arb(float(design[i, j]))*prior_chol[j][l] for j in range(k)), arb(0))
                      /arb(self.sigmas[i]) for l in range(k)] for i in range(n)]
                small = [[arb(int(i == j))+sum((row[i]*row[j] for row in t), arb(0))
                          for j in range(k)] for i in range(k)]
                factor = _cholesky(small)
                self.low_rank = t, factor
                self.log_normalization -= sum((arb(s).log() for s in self.sigmas), arb(0))
                self.log_normalization -= sum((factor[i][i].log() for i in range(k)), arb(0))
            elif covariance is None:
                self.log_normalization -= sum((arb(s).log() for s in self.sigmas), arb(0))
            else:
                c = np.diag(np.array(self.sigmas)**2) if covariance is None else np.asarray(covariance, float)
                if c.shape != (len(self.data), len(self.data)) or not np.array_equal(c, c.T):
                    raise ValueError("Exactly symmetric usable-pixel covariance required")
                n = len(c)
                # Diagonal variance is formed in Arb from the supplied sigma.
                c = [[arb(float(c[i, j])) if covariance is not None else
                      (arb(self.sigmas[i])*arb(self.sigmas[i]) if i == j else arb(0))
                      for j in range(n)] for i in range(n)]
                if linear is not None:
                    design = np.asarray(linear["design"], float)
                    mean = np.asarray(linear["mean"], float)
                    cov = np.asarray(linear["covariance"], float)
                    if (mean.ndim != 1 or design.shape != (n, len(mean))
                            or cov.shape != (len(mean), len(mean)) or not np.array_equal(cov, cov.T)
                            or not all(np.all(np.isfinite(x)) for x in [design, mean, cov])):
                        raise ValueError("Valid shared Gaussian linear nuisance required")
                    # Validate the supplied nuisance prior independently.
                    eigen_guard = np.linalg.cholesky(cov)
                    if not np.all(np.isfinite(eigen_guard)):
                        raise ValueError("Invalid nuisance covariance")
                    # Definiteness of the marginal is certified below. The
                    # nuisance covariance receives its own Arb LDL check.
                    ld = [[arb(float(cov[i, j])) for j in range(len(mean))] for i in range(len(mean))]
                    for k in range(len(mean)):
                        if not ld[k][k] > 0:
                            raise ValueError("Nuisance covariance not certified positive definite")
                        for i in range(k+1, len(mean)):
                            for j in range(k+1, len(mean)):
                                ld[i][j] -= ld[i][k]*ld[k][j]/ld[k][k]
                    b = [[arb(float(x)) for x in row] for row in design]
                    s = [[arb(float(x)) for x in row] for row in cov]
                    self.data = [arb(y)-sum((b[i][k]*arb(float(mean[k])) for k in range(len(mean))), arb(0))
                                 for i, y in enumerate(self.data)]
                    for i in range(n):
                        for j in range(n):
                            c[i][j] += sum((b[i][k]*s[k][l]*b[j][l]
                                          for k in range(len(mean)) for l in range(len(mean))), arb(0))
                self.cholesky = [[arb(0) for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(i+1):
                        value = c[i][j]-sum(
                            (self.cholesky[i][k]*self.cholesky[j][k] for k in range(j)), arb(0))
                        if i == j:
                            if not value > 0:
                                raise ValueError("Covariance not certified positive definite")
                            self.cholesky[i][j] = value.sqrt()
                            self.log_normalization -= self.cholesky[i][j].log()
                        else:
                            self.cholesky[i][j] = value/self.cholesky[j][j]
        else:
            # The scaled likelihood is bounded by one using each Poisson MLE.
            for y in self.data:
                k = int(y)
                self.log_normalization += (arb(k)*arb(k).log()-k if k else arb(0))-arb(factorial(k)).log()

    def roots(self):
        """Multinomial class counts, with IID continuous priors within each class.

        Integrating all labeled coordinates is correct with these multinomial
        weights. Symmetric queries see each physical unordered catalogue once
        in probability; an additional K! would be erroneous.
        """
        prior = self.manifest["count_prior"]
        if "poisson_mean" in prior:
            mean = arb(float(prior["poisson_mean"]))
            maximum = prior["max_count"]
            if not mean >= 0 or not isinstance(maximum, int) or not 0 <= maximum <= 12:
                raise ValueError("Nonnegative Poisson mean and max_count 0..12 required")
            probabilities = [(-mean).exp()]
            for k in range(1, maximum+1):
                probabilities.append(probabilities[-1]*mean/k)
            self.tail = (arb(1)-sum(probabilities, arb(0))).nonnegative_part()
        else:
            raw = prior["probabilities"]
            if not raw or any(not _finite(p) or p < 0 for p in raw) or sum(raw) <= 0:
                raise ValueError("Nonnegative source-count weights required")
            total = sum((arb(float(p)) for p in raw), arb(0))
            probabilities = [arb(float(p))/total for p in raw]
            self.tail = arb(0)
        type_weights = [arb(float(t.get("weight", 1.))) for t in self.types]
        if any(not x > 0 for x in type_weights):
            raise ValueError("Positive source-class weights required")
        total = sum(type_weights, arb(0))
        type_weights = [x/total for x in type_weights]
        roots = []
        for k, probability in enumerate(probabilities):
            if probability == 0:
                continue
            for classes in combinations_with_replacement(range(len(self.types)), k):
                counts = [classes.count(t) for t in range(len(self.types))]
                mass = probability*factorial(k)
                for count, weight in zip(counts, type_weights):
                    mass *= weight**count/factorial(count)
                names, ranges, fixed = [], [], {}
                for prefix, params in [("instrument", self.nuisance)]+[
                        (f"s{i}", self.types[t]["parameters"]) for i, t in enumerate(classes)]:
                    for key, (lo, hi) in sorted(params.items()):
                        name = f"{prefix}.{key}"
                        if lo == hi:
                            fixed[name] = float(lo)
                        else:
                            names.append(name)
                            ranges.append([float(lo), float(hi)])
                roots.append({"classes": list(classes), "names": names, "ranges": ranges,
                              "fixed": fixed, "mass": mass})
        return roots

    def means(self, root, parameters, *, source_only=None, source_fluxes=None):
        values = {**{k: arb(v) for k, v in root["fixed"].items()},
                  **dict(zip(root["names"], parameters))}
        nuisance = {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith("instrument.")}
        sources = []
        for index, cls in enumerate(root["classes"]):
            prefix = f"s{index}."
            sources.append((self.types[cls], {k[len(prefix):]: v for k, v in values.items()
                                             if k.startswith(prefix)}))
        result = []
        for exposure, active in zip(self.exposures, self.active):
            camera = exposure.get("instrument", "default")
            dt = arb(float(exposure["time"]))
            shape = np.asarray(exposure["data"]).shape
            get = lambda name, default=0: nuisance.get(f"{camera}_{name}", arb(default))
            dx, dy, background = get("dx"), get("dy"), get("background", exposure.get("background", 0))
            gain = get("log_gain").exp()
            psf = arb(float(exposure["psf_sigma"]))*get("log_psf").exp()
            scale = arb(float(exposure.get("flux_scale", 1.)))
            band = arb(float(exposure.get("band_coordinate", 0.)))
            pixels = [background if source_only is None else arb(0) for _ in active]
            for source_index, (source, p) in enumerate(sources):
                if source_only is not None and source_index != source_only:
                    continue
                if source["kind"] == "kepler":
                    from research.detection_limit.interval_orbits import apparent
                    state = [p[k] for k in ("rx", "ry", "rz", "vx", "vy", "vz")]
                    if any(isinstance(v, Jet) and any(c != 0 for c in v.c[1:]) for v in state):
                        raise NotImplementedError("Kepler geometry uses interval integration, not directional jets")
                    state = [_constant(v) for v in state]
                    fixed_orbit = all(source["parameters"][k][0] == source["parameters"][k][1]
                                      for k in ("rx", "ry", "rz", "vx", "vy", "vz"))
                    key = source["name"], exposure["id"]
                    if fixed_orbit and key in self._fixed_orbit_cache:
                        position = self._fixed_orbit_cache[key]
                    else:
                        position, _ = apparent(state, float(exposure["time"]), exposure["observer"],
                                               light_time_max=exposure.get("light_time_max", 1.))
                        if fixed_orbit:
                            self._fixed_orbit_cache[key] = position
                    direction = [x-arb(float(o)) for x, o in zip(position, exposure["observer"])]
                    dot = lambda v: sum((a*arb(float(b)) for a, b in zip(direction, v)), arb(0))
                    denominator = dot(exposure["center"])
                    if not denominator > 0:
                        raise ValueError("Orbit interval crosses tangent horizon")
                    angular_scale = arb(float(exposure["pixel_scale_arcsec"]))*arb.pi()/648000
                    x, y = dot(exposure["east"])/denominator/angular_scale, dot(exposure["north"])/denominator/angular_scale
                else:
                    x = p["x"]+p.get("vx", arb(0))*dt+p.get("parallax", arb(0))*arb(float(exposure.get("parallax_x", 0)))
                    y = p["y"]+p.get("vy", arb(0))*dt+p.get("parallax", arb(0))*arb(float(exposure.get("parallax_y", 0)))
                matrix = exposure.get("affine", [[1, 0], [0, 1]])
                shift = exposure.get("offset", [0, 0])
                x, y = (arb(float(matrix[0][0]))*x+arb(float(matrix[0][1]))*y+arb(float(shift[0]))+dx,
                        arb(float(matrix[1][0]))*x+arb(float(matrix[1][1]))*y+arb(float(shift[1]))+dy)
                sx, sy = psf, psf
                if source["kind"] == "galaxy":
                    # Sizes are explicitly in each detector's axes/pixel units.
                    sx = (square(psf)+square(p.get("sigma_x", arb(0)))).sqrt()
                    sy = (square(psf)+square(p.get("sigma_y", arb(0)))).sqrt()
                flux = p["flux"] if source_fluxes is None or source_index not in source_fluxes else source_fluxes[source_index]
                amplitude = flux*gain*scale*(p.get("color", arb(0))*band+p.get("variability", arb(0))*dt).exp()
                rx, ry = sx*arb(2).sqrt(), sy*arb(2).sqrt()
                for index, (j, i) in enumerate(active):
                    cx, cy = arb(2*i-shape[1]+1)/2, arb(2*j-shape[0]+1)/2
                    px = (((cx+arb(".5")-x)/rx).erf()-((cx-arb(".5")-x)/rx).erf())/2
                    py = (((cy+arb(".5")-y)/ry).erf()-((cy-arb(".5")-y)/ry).erf())/2
                    # CDF subtraction is interval-safe but can have a negative
                    # lower endpoint. Clipping only zeroth coefficients is safe.
                    if isinstance(px, Jet):
                        px.c[0] = px.c[0].nonnegative_part()
                        py.c[0] = py.c[0].nonnegative_part()
                    else:
                        px, py = px.nonnegative_part(), py.nonnegative_part()
                    pixels[index] = pixels[index]+amplitude*px*py
            result.extend(pixels)
        return result

    def _whiten(self, residuals):
        if self.cholesky is None:
            return [r/arb(s) for r, s in zip(residuals, self.sigmas)]
        whitened = []
        for i, r in enumerate(residuals):
            whitened.append((r-sum((self.cholesky[i][j]*whitened[j] for j in range(i)), arb(0)))
                            /self.cholesky[i][i])
        return whitened

    def _projection(self, values):
        t, factor = self.low_rank
        projected = []
        for j in range(len(factor)):
            v = sum((row[j]*x for row, x in zip(t, values)), arb(0))
            projected.append((v-sum((factor[j][i]*projected[i] for i in range(j)), arb(0)))/factor[j][j])
        return projected

    def _inner(self, left, right):
        value = sum((a*b for a, b in zip(left, right)), arb(0))
        if self.low_rank is not None:
            value -= sum((a*b for a, b in zip(self._projection(left), self._projection(right))), arb(0))
        return value

    def _norm(self, values):
        value = sum((square(v) for v in values), arb(0))
        if self.low_rank is not None:
            value -= sum((square(v) for v in self._projection(values)), arb(0))
            if isinstance(value, Jet):
                value.c[0] = value.c[0].nonnegative_part()
            else:
                value = value.nonnegative_part()
        return value

    def marginal_flux(self, root, parameters, axis, endpoints):
        """Exactly average one uniform flux, retaining all other sources.

        This is Gaussian completion of the square, not profiling the flux.
        It is also valid with marginalized shared linear Gaussian nuisance.
        """
        p0, p1 = list(parameters), list(parameters)
        p0[axis], p1[axis] = arb(0), arb(1)
        source_index = int(root["names"][axis].split(".")[0][1:])
        base = self.means(root, p0)
        response = self.means(root, p1, source_only=source_index)
        residual = self._whiten([arb(y)-b for y, b in zip(self.data, base)])
        template = self._whiten(response)
        information = self._norm(template)
        if not _constant(information) > 0:
            raise ValueError("Flux information not bounded away from zero")
        score = self._inner(template, residual)
        # Algebraically equal to C-B²/A, but preserves cancellation in
        # directional derivatives (especially collinear/blended sources).
        perpendicular = [r-score/information*p for r, p in zip(residual, template)]
        remainder = self._norm(perpendicular)
        lo, hi = endpoints
        width = (2*information).sqrt()
        cdf = ((information*hi-score)/width).erf()-((information*lo-score)/width).erf()
        if isinstance(cdf, Jet):
            cdf.c[0] = cdf.c[0].nonnegative_part()
        else:
            cdf = cdf.nonnegative_part()
        return (-remainder/2).exp()*(arb.pi()/(2*information)).sqrt()*cdf/(hi-lo)

    def likelihood(self, root, parameters):
        means = self.means(root, parameters)
        if self.noise == "gaussian":
            whitened = self._whiten([arb(y)-m for y, m in zip(self.data, means)])
            return (-self._norm(whitened)/2).exp()
        likelihood = arb(1)
        for y, mean in zip(self.data, means):
            if not _constant(mean) >= 0:
                raise ValueError("Poisson mean must be nonnegative throughout the prior")
            if y:
                # Polynomial form also covers zero intensity without log(0).
                # At large counts/dynamic range use the equivalent log form.
                if _constant(mean) > 0:
                    likelihood *= (arb(y)*(mean/arb(y)).log()-mean+y).exp()
                else:
                    likelihood *= (mean/arb(y))**int(y)*(-mean+y).exp()
            else:
                likelihood *= (-mean).exp()
        return likelihood

    def query(self, root, name, box=None):
        """Symmetric catalogue events; never queries an exchangeable label."""
        if name.startswith("count="):
            return len(root["classes"]) == int(name.split("=")[1])
        if name.startswith("has:"):
            return any(self.types[t]["name"] == name[4:] for t in root["classes"])
        if name.startswith("any:"):
            _, classname, parameter, lo, hi = name.split(":")
            lo, hi = float(lo), float(hi)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
                raise ValueError("Finite ordered query bounds required")
            intervals = dict(zip(root["names"], box if box is not None else root["ranges"]))
            unknown = False
            for i, cls in enumerate(root["classes"]):
                if self.types[cls]["name"] != classname:
                    continue
                key = f"s{i}.{parameter}"
                if key in root["fixed"]:
                    a = b = arb(root["fixed"][key])
                elif key in intervals:
                    a, b = map(arb, intervals[key])
                else:
                    raise ValueError("Query parameter absent from source class")
                if a >= lo and b <= hi:
                    return True
                if not (b < lo or a > hi):
                    unknown = True
            return None if unknown else False
        raise ValueError("Queries are count=K, has:class, or any:class:parameter:lo:hi")
