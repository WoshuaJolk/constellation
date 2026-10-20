"""Sequential variable-count catalogue inference.

Particles are an approximation, not a certificate of posterior mode coverage.
Every population starts from the complete declared prior. Rejuvenation targets
the joint likelihood of all assimilated exposures, retaining shared nuisance.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtr, xlogy, gammaln

from .scene import Scene


PARAMETERS = ("x", "y", "flux", "vx", "vy", "parallax", "variability",
              "color", "sigma_x", "sigma_y")
PARAMETER_INDEX = {name: i for i, name in enumerate(PARAMETERS)}
SCHEMA = "fit-sky-sequential-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def reflect(x):
    """Symmetric reflected random-walk proposal on the unit interval."""
    return 1-np.abs(np.remainder(x, 2)-1)


@dataclass
class Population:
    """Ordered IID source slots; inactive coordinates are auxiliary uniforms."""

    counts: np.ndarray
    classes: np.ndarray
    unit: np.ndarray
    nuisance: np.ndarray

    def take(self, rows):
        return Population(*(getattr(self, name)[rows].copy()
                            for name in ("counts", "classes", "unit", "nuisance")))

    def copy(self):
        return self.take(slice(None))

    def capacity(self, needed, rng):
        if needed <= self.classes.shape[1]:
            return
        extra = needed-self.classes.shape[1]
        self.classes = np.pad(self.classes, ((0, 0), (0, extra)), constant_values=-1)
        self.unit = np.concatenate([self.unit, rng.random((len(self.counts), extra,
                                                          len(PARAMETERS)))], axis=1)

    def validate(self, model):
        n, capacity = self.classes.shape
        if (n < 2 or self.counts.shape != (n,) or self.unit.shape != (n, capacity, len(PARAMETERS))
                or self.nuisance.shape != (n, len(model.nuisance_names))
                or np.any(self.counts < 0) or np.any(self.counts > capacity)):
            raise ValueError("Invalid particle shapes or counts")
        active = np.arange(capacity)[None, :] < self.counts[:, None]
        if (not np.issubdtype(self.counts.dtype, np.integer)
                or not np.issubdtype(self.classes.dtype, np.integer)
                or np.any(self.classes[active] < 0)
                or np.any(self.classes[active] >= len(model.scene.types))
                or np.any(self.classes[~active] != -1)
                or not np.all(np.isfinite(self.unit)) or not np.all(np.isfinite(self.nuisance))
                or np.any(self.unit < 0) or np.any(self.unit > 1)
                or np.any(self.nuisance < 0) or np.any(self.nuisance > 1)):
            raise ValueError("Invalid particle values")
        if np.any(~np.isfinite(model.log_count_prior(self.counts))):
            raise ValueError("Particle count outside prior support")


class BlockModel:
    """Joint likelihood with O(particles * pixel_block) rendering workspace.

    Images and metadata remain in memory. Dense arbitrary pixel covariance is
    unsupported here; diagonal noise plus shared Gaussian linear nuisance is
    handled by accumulated Woodbury statistics, without an Npixel² matrix.
    """

    def __init__(self, manifest, pixel_block=128):
        if not isinstance(pixel_block, int) or pixel_block < 1:
            raise ValueError("Positive integer pixel block required")
        if "covariance" in manifest:
            raise ValueError("Sequential blocks require diagonal noise; use linear_nuisance for shared covariance")
        self.scene = scene = Scene(manifest)
        if any(t["kind"] == "kepler" for t in scene.types):
            raise ValueError("Sequential renderer does not yet implement Kepler propagation")
        self.manifest = deepcopy(manifest)
        self.pixel_block = pixel_block
        self.nuisance_names = sorted(scene.nuisance)
        self.nuisance_bounds = np.array([scene.nuisance[k] for k in self.nuisance_names]).reshape(-1, 2)
        self.nuisance_index = {k: i for i, k in enumerate(self.nuisance_names)}
        self.bounds = np.array([[t["parameters"].get(k, [0., 0.]) for k in PARAMETERS]
                                for t in scene.types])
        self.extended = np.array([t["kind"] == "galaxy" for t in scene.types])
        self.type_weights = np.array([t.get("weight", 1.) for t in scene.types], float)
        if not np.all(np.isfinite(self.type_weights)) or np.any(self.type_weights <= 0):
            raise ValueError("Finite positive class weights required")
        self.type_weights /= self.type_weights.sum()
        prior = manifest["count_prior"]
        if "poisson_mean" in prior:
            self.poisson_mean = float(prior["poisson_mean"])
            if not np.isfinite(self.poisson_mean) or self.poisson_mean < 0:
                raise ValueError("Finite nonnegative Poisson mean required")
            self.count_probabilities = None
        else:
            weights = np.asarray(prior["probabilities"], float)
            if (weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights))
                    or np.any(weights < 0) or not weights.sum() > 0):
                raise ValueError("Nonnegative finite source-count weights required")
            self.count_probabilities = weights/weights.sum()
            self.poisson_mean = None
        self.ends = np.r_[0, np.cumsum([len(a) for a in scene.active])]
        self.data = np.array(list(map(float, scene.data)))
        self.sigma = np.array(scene.sigmas)
        self.linear = None if scene.low_rank is None else np.array(
            [[float(x) for x in row] for row in scene.low_rank[0]])
        self.cameras = sorted({e.get("instrument", "default") for e in scene.exposures})
        geometry = {k: [] for k in ["xx", "yy", "time", "px", "py", "psf", "gain",
                                    "band", "background", "camera", "a00", "a01", "a10",
                                    "a11", "ox", "oy"]}
        for e, active in zip(scene.exposures, scene.active):
            shape = np.shape(e["data"])
            a, off = e.get("affine", [[1, 0], [0, 1]]), e.get("offset", [0, 0])
            constants = dict(time=e["time"], px=e.get("parallax_x", 0),
                             py=e.get("parallax_y", 0), psf=e["psf_sigma"],
                             gain=e.get("flux_scale", 1), band=e.get("band_coordinate", 0),
                             background=e.get("background", 0),
                             camera=self.cameras.index(e.get("instrument", "default")),
                             a00=a[0][0], a01=a[0][1], a10=a[1][0], a11=a[1][1],
                             ox=off[0], oy=off[1])
            geometry["xx"].extend(i-(shape[1]-1)/2 for j, i in active)
            geometry["yy"].extend(j-(shape[0]-1)/2 for j, i in active)
            for key, value in constants.items():
                geometry[key].extend([value]*len(active))
        self.geometry = {k: np.asarray(v) for k, v in geometry.items()}
        if not all(np.all(np.isfinite(v)) for v in self.geometry.values()):
            raise ValueError("Finite detector geometry required")
        self.factors, self.normalization = [None], [0.]
        rank = 0 if self.linear is None else self.linear.shape[1]
        gram = np.eye(rank)
        norm = 0.
        for j in range(len(scene.exposures)):
            lo, hi = self.ends[j:j+2]
            if scene.noise == "gaussian":
                norm -= .5*(hi-lo)*np.log(2*np.pi)+np.log(self.sigma[lo:hi]).sum()
                factor = None
                if rank:
                    gram += self.linear[lo:hi].T@self.linear[lo:hi]
                    factor = np.linalg.cholesky(gram)
                self.factors.append(factor)
                self.normalization.append(float(norm-(0 if factor is None else np.log(np.diag(factor)).sum())))
            else:
                y = self.data[lo:hi]
                norm += np.sum(xlogy(y, y)-y-gammaln(y+1))
                self.factors.append(None)
                self.normalization.append(float(norm))
        self.evaluated_catalogues = 0
        self.evaluated_catalogue_pixels = 0

    def log_count_prior(self, counts):
        counts = np.asarray(counts)
        if self.count_probabilities is None:
            return xlogy(counts, self.poisson_mean)-self.poisson_mean-gammaln(counts+1)
        result = np.full(counts.shape, -np.inf)
        valid = (counts >= 0) & (counts < len(self.count_probabilities))
        with np.errstate(divide="ignore"):
            result[valid] = np.log(self.count_probabilities[counts[valid]])
        return result

    def prior(self, size, rng):
        counts = (rng.poisson(self.poisson_mean, size) if self.count_probabilities is None else
                  rng.choice(len(self.count_probabilities), size, p=self.count_probabilities))
        capacity = max(1, int(counts.max())+1)
        classes = rng.choice(len(self.type_weights), (size, capacity), p=self.type_weights)
        classes[np.arange(capacity)[None, :] >= counts[:, None]] = -1
        return Population(counts, classes, rng.random((size, capacity, len(PARAMETERS))),
                          rng.random((size, len(self.nuisance_names))))

    def physical(self, population):
        indices = np.maximum(population.classes, 0)
        bounds = self.bounds[indices]
        sources = bounds[..., 0]+population.unit*(bounds[..., 1]-bounds[..., 0])
        b = self.nuisance_bounds
        nuisance = b[:, 0]+population.nuisance*(b[:, 1]-b[:, 0])
        return sources, nuisance

    def _means(self, population, sources, nuisance, lo, hi):
        g = {k: v[lo:hi] for k, v in self.geometry.items()}
        n = len(population.counts)

        def instrument(key, default=0.):
            out = np.broadcast_to(np.asarray(default, float), (n, hi-lo)).copy()
            for camera in self.cameras:
                column = self.nuisance_index.get(camera+"_"+key)
                if column is not None:
                    mask = g["camera"] == self.cameras.index(camera)
                    out[:, mask] = nuisance[:, column, None]
            return out

        means = instrument("background", g["background"])
        dx, dy = instrument("dx"), instrument("dy")
        psf = g["psf"]*np.exp(instrument("log_psf"))
        gain = g["gain"]*np.exp(instrument("log_gain"))
        for slot in range(int(population.counts.max())):
            rows = np.flatnonzero(population.counts > slot)
            p = {name: sources[rows, slot, j, None] for j, name in enumerate(PARAMETERS)}
            x = p["x"]+p["vx"]*g["time"]+p["parallax"]*g["px"]
            y = p["y"]+p["vy"]*g["time"]+p["parallax"]*g["py"]
            x, y = g["a00"]*x+g["a01"]*y+g["ox"]+dx[rows], g["a10"]*x+g["a11"]*y+g["oy"]+dy[rows]
            extended = self.extended[population.classes[rows, slot], None]
            sx = np.hypot(psf[rows], np.where(extended, p["sigma_x"], 0.))
            sy = np.hypot(psf[rows], np.where(extended, p["sigma_y"], 0.))
            flux = p["flux"]*gain[rows]*np.exp(p["color"]*g["band"]+p["variability"]*g["time"])
            means[rows] += flux*(ndtr((g["xx"]+.5-x)/sx)-ndtr((g["xx"]-.5-x)/sx))*(
                ndtr((g["yy"]+.5-y)/sy)-ndtr((g["yy"]-.5-y)/sy))
        if not np.all(np.isfinite(means)):
            raise ArithmeticError("Nonfinite sequential forward model")
        return means

    def evaluate(self, population, endpoints):
        """Normalized joint log likelihoods at exposure-prefix endpoints."""
        if any(not isinstance(x, (int, np.integer)) or x < 0 or x > len(self.scene.exposures)
               for x in endpoints):
            raise ValueError("Exposure prefix outside manifest")
        unique = sorted(set(endpoints))
        n = len(population.counts)
        sources, nuisance = self.physical(population)
        rank = 0 if self.linear is None else self.linear.shape[1]
        energy, score = np.zeros(n), np.zeros((n, rank))
        result, last = {}, 0
        for endpoint in unique:
            end = self.ends[endpoint]
            for lo in range(last, end, self.pixel_block):
                hi = min(end, lo+self.pixel_block)
                means = self._means(population, sources, nuisance, lo, hi)
                if self.scene.noise == "gaussian":
                    r = (self.data[lo:hi]-means)/self.sigma[lo:hi]
                    energy += np.sum(r*r, axis=1)
                    if rank:
                        score += r@self.linear[lo:hi]
                else:
                    if np.any(means < 0):
                        raise ValueError("Negative Poisson intensity")
                    y = self.data[lo:hi]
                    energy += -2*np.sum(xlogy(y, means)-means-xlogy(y, y)+y, axis=1)
            remainder = energy.copy()
            if rank and endpoint:
                projected = solve_triangular(self.factors[endpoint], score.T, lower=True).T
                remainder -= np.sum(projected*projected, axis=1)
                if np.any(remainder < -1e-8*np.maximum(1, energy)):
                    raise ArithmeticError("Negative marginalized Gaussian energy")
                remainder = np.maximum(remainder, 0.)
            result[endpoint] = self.normalization[endpoint]-.5*remainder
            last = end
        self.evaluated_catalogues += n
        self.evaluated_catalogue_pixels += n*last
        return np.array([result[e] for e in endpoints]).T

    def events(self, population, queries):
        sources, _ = self.physical(population)
        active = np.arange(population.classes.shape[1])[None, :] < population.counts[:, None]
        names = [t["name"] for t in self.scene.types]
        result = {}
        for query in queries:
            if query.startswith("count="):
                result[query] = population.counts == int(query[6:])
            elif query.startswith("has:"):
                cls = names.index(query[4:])
                result[query] = np.any(active & (population.classes == cls), axis=1)
            elif query.startswith("any:"):
                _, cls, param, lo, hi = query.split(":")
                lo, hi = float(lo), float(hi)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
                    raise ValueError("Finite ordered event bounds required")
                index = names.index(cls)
                if param not in self.scene.types[index]["parameters"]:
                    raise ValueError("Query parameter absent from source class")
                v = sources[..., PARAMETER_INDEX[param]]
                result[query] = np.any(active & (population.classes == index)
                                       & (v >= lo) & (v <= hi), axis=1)
            else:
                raise ValueError(f"Unsupported sequential query: {query}")
        return result

    def catalogues(self, population):
        sources, nuisance = self.physical(population)
        result = []
        for row, count in enumerate(population.counts):
            catalogue = []
            for slot in range(count):
                t = self.scene.types[population.classes[row, slot]]
                catalogue.append({"class": t["name"], "kind": t["kind"], "parameters": {
                    key: float(sources[row, slot, PARAMETER_INDEX[key]]) for key in t["parameters"]}})
            result.append({"sources": catalogue, "nuisance": dict(zip(self.nuisance_names, nuisance[row].tolist()))})
        return result


def propose_counts(model, population, rng):
    """Prior-drawn births with uniform insertion, and uniform-source deaths.

    Fixed 1/2 birth and 1/2 death probabilities, including a self-transition for
    an empty death, avoid a hidden boundary correction. Forward/reverse source
    densities and insertion/deletion probabilities cancel, leaving p(K')/p(K).
    """
    proposal = population.copy()
    n = len(population.counts)
    birth = rng.random(n) < .5
    proposed_counts = population.counts+np.where(birth, 1, -1)
    viable = (proposed_counts >= 0) & np.isfinite(model.log_count_prior(np.maximum(proposed_counts, 0)))
    proposed_counts = np.where(viable, proposed_counts, population.counts)
    proposal.capacity(max(1, int(proposed_counts.max())+1), rng)
    for row in np.flatnonzero(viable):
        k = int(population.counts[row])
        if birth[row]:
            slot = int(rng.integers(k+1))
            proposal.classes[row, slot+1:k+1] = proposal.classes[row, slot:k].copy()
            proposal.unit[row, slot+1:k+1] = proposal.unit[row, slot:k].copy()
            proposal.classes[row, slot] = rng.choice(len(model.type_weights), p=model.type_weights)
            proposal.unit[row, slot] = rng.random(len(PARAMETERS))
        else:
            slot = int(rng.integers(k))
            proposal.classes[row, slot:k-1] = proposal.classes[row, slot+1:k].copy()
            proposal.unit[row, slot:k-1] = proposal.unit[row, slot+1:k].copy()
            proposal.classes[row, k-1] = -1
            proposal.unit[row, k-1] = rng.random(len(PARAMETERS))
    proposal.counts = proposed_counts
    ratio = model.log_count_prior(proposed_counts)-model.log_count_prior(population.counts)
    # A fixed-probability whole-catalogue refresh connects finite count priors
    # with holes and preserves a full-support escape from any local catalogue.
    # All source/class/count proposal factors cancel the corresponding priors.
    refresh = rng.random(n) < .05
    if np.any(refresh):
        independent = model.prior(int(refresh.sum()), rng)
        proposal.capacity(independent.classes.shape[1], rng)
        independent.capacity(proposal.classes.shape[1], rng)
        proposal.counts[refresh] = independent.counts
        proposal.classes[refresh] = independent.classes
        proposal.unit[refresh] = independent.unit
        ratio[refresh] = 0.
        viable[refresh] = True
    return proposal, ratio, viable


def propose_parameters(model, population, rng, scale):
    """Fixed-mixture symmetric moves, including full-support prior refreshes."""
    proposal = population.copy()
    n = len(population.counts)
    which = rng.random(n)
    for row in range(n):
        k = int(population.counts[row])
        if which[row] < .2 and len(model.nuisance_names):
            proposal.nuisance[row] = reflect(population.nuisance[row]+rng.normal(
                0, scale, len(model.nuisance_names)))
        elif k:
            slot = int(rng.integers(k))
            if which[row] > .95:
                proposal.classes[row, slot] = rng.choice(len(model.type_weights), p=model.type_weights)
                proposal.unit[row, slot] = rng.random(len(PARAMETERS))
            elif which[row] > .8:
                # Common unit coordinates give a dimension-preserving auxiliary
                # representation of a class change. Class-prior proposals cancel.
                proposal.classes[row, slot] = rng.choice(len(model.type_weights), p=model.type_weights)
            elif which[row] > .5:
                cls = population.classes[row, slot]
                axes = np.flatnonzero(model.bounds[cls, :, 1] > model.bounds[cls, :, 0])
                if len(axes):
                    axis = rng.choice(axes)
                    proposal.unit[row, slot, axis] = reflect(
                        population.unit[row, slot, axis]+rng.normal(0, scale))
            else:
                proposal.unit[row, slot] = reflect(population.unit[row, slot]+rng.normal(
                    0, scale/np.sqrt(3), len(PARAMETERS)))
        elif len(model.nuisance_names):
            proposal.nuisance[row] = reflect(population.nuisance[row]+rng.normal(
                0, scale, len(model.nuisance_names)))
    return proposal, np.zeros(n), np.ones(n, bool)


def conditional_ess(log_weights, increment, delta):
    if delta == 0:
        return float(len(log_weights))
    values = delta*increment
    return float(len(log_weights)*np.exp(2*logsumexp(log_weights+values)
                                        -logsumexp(log_weights+2*values)))


def next_temperature(log_weights, increment, beta, fraction):
    if not np.all(np.isfinite(increment)):
        raise ArithmeticError("Nonfinite tempering increments; the current particle support is insufficient")
    remaining = 1-beta
    target = fraction*len(log_weights)
    if conditional_ess(log_weights, increment, remaining) >= target:
        return 1.
    lo, hi = 0., remaining
    for _ in range(48):
        mid = .5*(lo+hi)
        if conditional_ess(log_weights, increment, mid) < target:
            hi = mid
        else:
            lo = mid
    if lo <= 1e-12:
        raise ArithmeticError("Tempering stalled; no forced step may hide particle collapse")
    return beta+lo


def systematic_resample(log_weights, rng):
    weights = np.exp(log_weights-logsumexp(log_weights))
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative, (rng.random()+np.arange(len(weights)))/len(weights))


def _accept(population, proposal, likelihood, proposed_likelihood, ratio, beta, rng):
    delta = ((1-beta)*(proposed_likelihood[:, 0]-likelihood[:, 0])
             +beta*(proposed_likelihood[:, 1]-likelihood[:, 1])+ratio)
    if np.any(np.isnan(delta)):
        raise ArithmeticError("Undefined Metropolis ratio")
    accept = np.log(rng.random(len(delta))) < np.minimum(0., delta)
    population.capacity(proposal.classes.shape[1], rng)
    population.counts[accept] = proposal.counts[accept]
    population.classes[accept] = proposal.classes[accept]
    population.unit[accept] = proposal.unit[accept]
    population.nuisance[accept] = proposal.nuisance[accept]
    likelihood[accept] = proposed_likelihood[accept]
    return accept


class SequentialFit:
    """Resumable approximate posterior population for a fixed generative model."""

    def __init__(self, manifest, *, particles=2048, seed=20261110, sweeps=20,
                 ess_fraction=.8, pixel_block=128, max_stages=1000, queries=None):
        if (not isinstance(particles, int) or particles < 2 or not isinstance(sweeps, int)
                or sweeps < 1 or not 0 < ess_fraction < 1 or not isinstance(max_stages, int)
                or max_stages < 1):
            raise ValueError("Valid particle, sweep, ESS and stage settings required")
        self.model = BlockModel(manifest, pixel_block)
        self.settings = dict(particles=particles, seed=seed, sweeps=sweeps,
                             ess_fraction=ess_fraction, pixel_block=pixel_block, max_stages=max_stages)
        self.rng = np.random.default_rng(seed)
        self.population = self.model.prior(particles, self.rng)
        self.queries = queries or ["has:"+t["name"] for t in self.model.scene.types]
        self.model.events(self.population, self.queries)
        self.log_weights = np.full(particles, -np.log(particles))
        self.log_evidence = 0.
        self.assimilated = 0
        self.stages, self.history = [], []
        self.seconds = 0.
        self.in_progress = False

    def assimilate(self, endpoint=None, *, progress=None):
        """Append a prefix, with likelihood-ratio bridges to the joint model."""
        endpoint = len(self.model.scene.exposures) if endpoint is None else endpoint
        if self.in_progress:
            raise RuntimeError("Previous assimilation was interrupted; reload the last completed checkpoint")
        if not isinstance(endpoint, int) or not self.assimilated < endpoint <= len(self.model.scene.exposures):
            raise ValueError("Assimilation must append new exposures exactly once")
        started = perf_counter()
        self.in_progress = True
        old = self.assimilated
        likelihood = self.model.evaluate(self.population, [old, endpoint])
        beta = 0.
        while beta < 1:
            if len(self.stages) >= self.settings["max_stages"]:
                raise RuntimeError("Sequential stage budget exhausted; no convergence result is available")
            previous = beta
            beta = next_temperature(self.log_weights, likelihood[:, 1]-likelihood[:, 0],
                                    previous, self.settings["ess_fraction"])
            unnormalized = self.log_weights+(beta-previous)*(likelihood[:, 1]-likelihood[:, 0])
            log_increment = float(logsumexp(unnormalized))
            self.log_evidence += log_increment
            self.log_weights = unnormalized-log_increment
            ess = float(np.exp(-logsumexp(2*self.log_weights)))
            indices = systematic_resample(self.log_weights, self.rng)
            self.population = self.population.take(indices)
            likelihood = likelihood[indices].copy()
            self.log_weights.fill(-np.log(len(indices)))
            counts_accepted = counts_proposed = continuous_accepted = 0
            # A fixed mixture of proposal scales avoids fitting kernels to the
            # same finite population and pretending that adaptation is exact.
            scales = (.025, .08, .25)
            for sweep in range(self.settings["sweeps"]):
                proposal, ratio, viable = propose_counts(self.model, self.population, self.rng)
                proposed = self.model.evaluate(proposal, [old, endpoint])
                accepted = _accept(self.population, proposal, likelihood, proposed, ratio, beta, self.rng)
                counts_accepted += int(np.sum(accepted & viable))
                counts_proposed += int(viable.sum())
                proposal, ratio, _ = propose_parameters(self.model, self.population, self.rng,
                                                        scales[sweep % len(scales)])
                proposed = self.model.evaluate(proposal, [old, endpoint])
                accepted = _accept(self.population, proposal, likelihood, proposed, ratio, beta, self.rng)
                continuous_accepted += int(accepted.sum())
            stage = {
                "from_exposures": old, "to_exposures": endpoint, "beta": beta,
                "ess_before_resampling": ess, "distinct_resampled_parents": int(len(np.unique(indices))),
                "log_evidence_increment_estimate": log_increment,
                "count_acceptance": counts_accepted/max(1, counts_proposed),
                "parameter_acceptance": continuous_accepted/(len(indices)*self.settings["sweeps"]),
                "maximum_sampled_count": int(self.population.counts.max()),
            }
            self.stages.append(stage)
            if progress:
                progress(stage)
        self.assimilated = endpoint
        self.in_progress = False
        self.seconds += perf_counter()-started
        summary = self.summary()
        self.history.append(summary)
        return summary

    def summary(self):
        weights = np.exp(self.log_weights)
        events = self.model.events(self.population, self.queries)
        counts = {str(k): float(weights[self.population.counts == k].sum())
                  for k in np.unique(self.population.counts)}
        _, nuisance = self.model.physical(self.population)
        return {
            "assimilated_exposures": self.assimilated,
            "log_evidence_estimate": self.log_evidence,
            "queries": {q: {"estimate": float(weights@v), "certified_interval": [0., 1.]}
                        for q, v in events.items()},
            "count_distribution_estimate": counts,
            "nuisance_mean_estimate": {name: float(weights@nuisance[:, j])
                                       for j, name in enumerate(self.model.nuisance_names)},
            "maximum_sampled_count": int(self.population.counts.max()),
            "stages": len(self.stages), "seconds": self.seconds,
        }

    def report(self, *, include_catalogues=False):
        if self.in_progress:
            raise RuntimeError("Cannot report an interrupted assimilation as a completed posterior")
        result = {
            "schema": SCHEMA, "status": "approximate_uncertified",
            "full_objective_complete": False, "accuracy_target_met": False,
            "manifest_sha256": digest(self.model.manifest), "settings": self.settings,
            "scope": "Float64 sequential Monte Carlo estimates; no finite-sample mode-coverage or evidence-error certificate.",
            "source_count_scope": "Complete Poisson prior, no enumeration cutoff" if self.model.count_probabilities is None
                                  else "Complete supplied finite count prior",
            "linear_nuisance_scope": "Shared Gaussian linear nuisance integrated jointly, not sampled",
            "summary": self.summary(), "history": self.history, "stages": self.stages,
            "work": {"catalogue_evaluations": self.model.evaluated_catalogues,
                     "catalogue_pixels": int(self.model.evaluated_catalogue_pixels),
                     "pixel_block": self.model.pixel_block},
        }
        if include_catalogues:
            result["catalogues"] = self.model.catalogues(self.population)
            result["weights"] = np.exp(self.log_weights).tolist()
        return result

    def save(self, path):
        """Save a non-pickle checkpoint atomically; completed assimilation only."""
        if self.in_progress:
            raise RuntimeError("Cannot checkpoint an interrupted assimilation; retain the previous checkpoint")
        path = Path(path)
        metadata = {
            "schema": SCHEMA, "manifest": self.model.manifest, "settings": self.settings,
            "queries": self.queries, "rng": self.rng.bit_generator.state,
            "assimilated": self.assimilated, "log_evidence": self.log_evidence,
            "stages": self.stages, "history": self.history, "seconds": self.seconds,
            "evaluated_catalogues": self.model.evaluated_catalogues,
            "evaluated_catalogue_pixels": int(self.model.evaluated_catalogue_pixels),
        }
        temp = path.with_name(path.name+".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, allow_nan=False)),
                                counts=self.population.counts, classes=self.population.classes,
                                unit=self.population.unit, nuisance=self.population.nuisance,
                                log_weights=self.log_weights)
        temp.replace(path)

    @classmethod
    def load(cls, path, manifest=None):
        """Resume unchanged data or an appended exposure sequence.

        All old observations, priors, instruments and existing linear-design
        rows must match exactly. Changed noise/data requires a fresh fit.
        """
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["schema"] != SCHEMA:
                raise ValueError("Unsupported sequential checkpoint")
            original = metadata["manifest"]
            manifest = deepcopy(original if manifest is None else manifest)
            nold = len(original["exposures"])
            if manifest["exposures"][:nold] != original["exposures"]:
                raise ValueError("Existing observations changed in checkpoint resume")
            old_config, new_config = deepcopy(original), deepcopy(manifest)
            old_config.pop("exposures")
            new_config.pop("exposures")
            if "linear_nuisance" in old_config and "linear_nuisance" in new_config:
                nrows = len(old_config["linear_nuisance"]["design"])
                new_design = new_config["linear_nuisance"]["design"]
                if new_design[:nrows] != old_config["linear_nuisance"]["design"]:
                    raise ValueError("Existing shared nuisance design changed")
                new_config["linear_nuisance"]["design"] = new_design[:nrows]
            if new_config != old_config:
                raise ValueError("Generative prior or instrument configuration changed")
            fit = cls(manifest, **metadata["settings"], queries=metadata["queries"])
            fit.population = Population(*(saved[k].copy() for k in ("counts", "classes", "unit", "nuisance")))
            fit.population.validate(fit.model)
            fit.log_weights = saved["log_weights"].copy()
            if (fit.log_weights.shape != (fit.settings["particles"],)
                    or len(fit.population.counts) != fit.settings["particles"]
                    or not np.all(np.isfinite(fit.log_weights))
                    or abs(logsumexp(fit.log_weights)) > 1e-10):
                raise ValueError("Invalid checkpoint particle weights")
        fit.rng.bit_generator.state = metadata["rng"]
        for key in ("assimilated", "log_evidence", "stages", "history", "seconds"):
            setattr(fit, key, metadata[key])
        if (not isinstance(fit.assimilated, int) or not 0 <= fit.assimilated <= nold
                or not np.isfinite(fit.log_evidence)):
            raise ValueError("Invalid checkpoint assimilation state")
        for key in ("evaluated_catalogues", "evaluated_catalogue_pixels"):
            setattr(fit.model, key, metadata[key])
        return fit
