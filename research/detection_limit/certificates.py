"""Primal/dual Gaussian-mixture certificates with continuous-model error.

These are applications of established testing duality and concentration bounds.
Certificates have stated Monte Carlo confidence, conditional on the supplied
model/cover. Ordinary NumPy/SciPy evaluation is not formal interval arithmetic.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, logsumexp
from scipy.stats import beta, binom, norm


def gaussian_tv(radius):
    """Exact TV distance for equal-covariance Gaussians, in whitened units."""
    radius = np.asarray(radius, float)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0):
        raise ValueError("Mean-error radii must be finite and nonnegative")
    return erf(radius / (2 * np.sqrt(2)))


def _weights(weights, count):
    weights = np.asarray(weights, float)
    if (weights.shape != (count,) or np.any(~np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Supply nonnegative weights with positive total")
    return weights / weights.sum()


@dataclass
class GaussianCover:
    """Every continuous signal in cell i is within radii[i] of means[i].

    The class does not establish coverage: constructors/callers must do that.
    Weights are exact cell probabilities for the prior-average interpretation.
    Every cell must be nonempty for the minimax interpretation. The means need
    not be physical members; their distance to every cell member is bounded.
    """

    means: np.ndarray
    radii: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.means = np.asarray(self.means, float)
        self.radii = np.asarray(self.radii, float)
        if (self.means.ndim != 2 or min(self.means.shape) == 0
                or np.any(~np.isfinite(self.means))):
            raise ValueError("Means must be a finite nonempty matrix")
        if self.radii.shape != (len(self.means),):
            raise ValueError("One covering radius is required per mean")
        gaussian_tv(self.radii)
        self.weights = _weights(self.weights, len(self.means))

    @property
    def tv(self):
        return gaussian_tv(self.radii)

    def compressed(self, residual_budget=1e-8):
        """Reduce score dimension, charging discarded signal norm to the cover.

        This is numerical SVD, not a validated orthogonality proof. No result
        silently discards a signal direction to make an upper bound easier.
        """
        if not np.isfinite(residual_budget) or residual_budget < 0:
            raise ValueError("Residual budget must be finite and nonnegative")
        u, s, vt = np.linalg.svd(self.means, full_matrices=False)
        for rank in range(1, len(s) + 1):
            coordinates = u[:, :rank] * s[:rank]
            residual = np.linalg.norm(self.means - coordinates @ vt[:rank], axis=1)
            if np.max(residual) <= residual_budget or rank == len(s):
                return GaussianCover(coordinates, self.radii + residual, self.weights), {
                    "input_dimension": self.means.shape[1], "score_dimension": rank,
                    "maximum_discarded_mean_norm": float(np.max(residual)),
                    "pixel_to_score_basis": vt[:rank].tolist(),
                    "arithmetic": "float64 SVD; not an interval orthogonality certificate",
                }


def log_lr(y, means, weights):
    """Mixture density / N(0,I) density, with batch observations."""
    y, means = np.asarray(y, float), np.asarray(means, float)
    weights = _weights(weights, len(means))
    with np.errstate(divide="ignore"):
        logs = np.log(weights)
    return logsumexp(y @ means.T - np.sum(means**2, axis=1) / 2 + logs, axis=-1)


def mixture_scores(means, weights, count, rng, component=None, batch_size=2048):
    """Independent draws, keeping only the one-dimensional LR scores."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Sample count must be a positive integer")
    out = np.empty(count)
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        labels = rng.choice(len(means), n, p=weights) if component is None else component
        y = rng.normal(size=(n, means.shape[1])) + means[labels]
        out[start:start+n] = log_lr(y, means, weights)
    return out


def choose_threshold(log_scores, alpha):
    """Minimize the empirical dual on an INDEPENDENT training sample from Q.

    No null-tail sampling: derivative of D(lambda) is
    alpha - E_Q[1{L>lambda}/L]. Training chooses a candidate, not a guarantee.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    scores = np.sort(np.asarray(log_scores, float))[::-1]
    if scores.ndim != 1 or not len(scores) or np.any(~np.isfinite(scores)):
        raise ValueError("Finite nonempty training scores required")
    cumulative = np.cumsum(np.exp(np.minimum(-scores, 700))) / len(scores)
    index = min(int(np.searchsorted(cumulative, alpha)), len(scores)-1)
    # lambda > 1/alpha has dual >= 1 and cannot improve on the trivial bound.
    return float(np.clip(scores[index], -700, -math.log(alpha)))


def _lower_binomial(successes, trials, failure):
    return 0.0 if successes == 0 else float(beta.ppf(failure, successes, trials-successes+1))


def choose_rule(scores, alpha):
    """Allow randomized boundary decisions, essential for discrete observations."""
    threshold = choose_threshold(scores,alpha)
    scores = np.asarray(scores,float)
    above = float(np.sum(np.exp(-scores[scores>threshold]))/len(scores))
    boundary = float(np.sum(scores==threshold)/len(scores))*math.exp(-threshold)
    tie_probability = float(np.clip((alpha-above)/boundary,0,1)) if boundary else 0.0
    return threshold,tie_probability


def _reject(scores, threshold, tie_probability, rng):
    return (scores>threshold) | ((scores==threshold) & (rng.random(len(scores))<tie_probability))


def _importance_bounds(scores, rejects, threshold, alpha_work, alpha_dual, failure):
    lam = math.exp(threshold)
    w = np.where(rejects,np.exp(np.minimum(threshold-scores,0)),0.0)
    h = -np.expm1(np.minimum(threshold-scores,0))
    def error(values):
        # Maurer & Pontil (2009), Theorem 4, unbiased sample variance.
        if len(values)<2:
            return 1.0
        log=math.log(2/failure)
        return min(1.0, math.sqrt(2*float(np.var(values,ddof=1))*log/len(values))
                   +7*log/(3*(len(values)-1)))
    slack={"null_importance":error(w),"dual_hinge":error(h)}
    null_upper = min(1.0,(float(np.mean(w))+slack["null_importance"])/lam)
    gamma = min(1.0,alpha_work/null_upper)
    dual = min(1.0,lam*alpha_dual+float(np.mean(h))+slack["dual_hinge"])
    return lam,w,h,slack,null_upper,gamma,dual


def certify_likelihood(draw_log_lr, *, alpha=0.01, confidence=0.99, samples=262144,
                       training_samples=16384, seed=2026, requested_gap=0.01):
    """Model-independent engine, GIVEN a correct alternative sampler and log LR.

    draw_log_lr(rng,count) must draw iid Y~P1 and return log(dP1/dP0)(Y).
    This interface does not solve continuous-orbit likelihood integration.
    It supports discrete data and atoms using randomized boundary decisions.
    """
    if not 0<alpha<1 or not 0<confidence<1 or not 0<requested_gap<1:
        raise ValueError("Probabilities/gap must lie in (0,1)")
    for n in (samples,training_samples):
        if not isinstance(n,int) or isinstance(n,bool) or n<1:
            raise ValueError("Positive integer sample counts required")
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    def draw(rng,n):
        values=np.asarray(draw_log_lr(rng,n),float)
        if values.shape!=(n,) or np.any(~np.isfinite(values)):
            raise ValueError("Sampler must return finite log LR values of requested length")
        return values
    threshold,tie=choose_rule(draw(rngs[0],training_samples),alpha)
    scores=draw(rngs[1],samples)
    rejects=_reject(scores,threshold,tie,rngs[1])
    failure=(1-confidence)/3
    lam,w,h,slack,null_upper,gamma,upper=_importance_bounds(scores,rejects,threshold,alpha,alpha,failure)
    lower=gamma*_lower_binomial(int(np.sum(rejects)),samples,failure)
    use_coin=lower<alpha
    lower=max(lower,alpha)
    if upper<lower-1e-12:
        raise ArithmeticError("Inconsistent certificate; investigate confidence event/model")
    return {"objective":"specified simple alternative", "alpha":alpha,
            "simultaneous_confidence":confidence,"optimal_power_interval":[lower,upper],
            "optimality_gap_upper":max(0,upper-lower),"gap_target_met":bool(upper-lower<=requested_gap),
            "requested_gap":requested_gap,
            "detector":{"kind":"constant_coin" if use_coin else "randomized_likelihood_ratio",
                        "log_lr_threshold":threshold,"boundary_acceptance_probability":tie,
                        "acceptance_probability_above_threshold":gamma,
                        "constant_coin_probability":alpha if use_coin else None},
            "null_probability_upper":alpha if use_coin else gamma*null_upper,
            "dual_lambda":lam,"bounded_expectation_slack":slack,
            "training_samples":training_samples,"validation_samples":samples,
            "arithmetic":"float64 LR/statistical evaluation; not a formal roundoff certificate"}


def certify(cover, *, alpha=0.01, confidence=0.99, samples=262144,
            training_samples=16384, seed=2026, objective="average",
            detector_weights=None, null_tv=0.0, alternative_tv=0.0,
            requested_gap=0.01, component_samples=None, calibration="alternative"):
    """Certify an executable randomized LR detector and an optimal-power bound.

    Independent training chooses lambda. Under Q, H=(1-lambda/L)+ and
    W=(lambda/L)1{L>lambda} are bounded by one. Empirical Bernstein bounds yield
    D <= lambda*alpha + E_Q H and P0(reject) <= E_Q W / lambda.
    This avoids estimating tiny false-alarm probabilities by null simulation.

    The returned detector rejects with probability gamma if log_lr > threshold.
    Seeds are reproducibility aids, not evidence that a model is true.
    """
    if objective not in {"average", "minimax"}:
        raise ValueError("objective must be average or minimax")
    if calibration not in {"alternative","null_order_statistic"}:
        raise ValueError("Unknown threshold calibration method")
    for count in (samples,training_samples):
        if not isinstance(count,int) or isinstance(count,bool) or count<1:
            raise ValueError("Positive integer sample counts required")
    if component_samples is not None and (not isinstance(component_samples,int)
                                          or isinstance(component_samples,bool) or component_samples<1):
        raise ValueError("Positive integer component sample count required")
    if not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError("alpha and confidence must lie in (0,1)")
    if not 0 <= null_tv < alpha or not 0 <= alternative_tv < 1:
        raise ValueError("Require 0 <= null TV < alpha and 0 <= alternative TV < 1")
    if not 0 < requested_gap < 1:
        raise ValueError("Requested gap must lie in (0,1)")
    if objective == "average" and detector_weights is not None:
        supplied = _weights(detector_weights, len(cover.means))
        if not np.allclose(supplied, cover.weights, rtol=0, atol=1e-14):
            raise ValueError("Average certificate requires the declared prior weights")
    weights = cover.weights if detector_weights is None else _weights(detector_weights, len(cover.means))
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    alpha_work = alpha - null_tv
    events = (3 if objective == "average" else 2 + len(cover.means)) + (calibration=="null_order_statistic")
    failure = (1-confidence) / events
    if calibration=="alternative":
        trained = mixture_scores(cover.means, weights, training_samples, generators[0])
        threshold,tie_probability = choose_rule(trained, alpha_work)
        calibration_order=None
    else:
        calibration_order=int(binom.ppf(1-failure,training_samples,1-alpha_work))+1
        if calibration_order>training_samples:
            raise ValueError("Too few null calibration samples for requested alpha/confidence")
        trained=[]
        for start in range(0,training_samples,2048):
            y=generators[0].normal(size=(min(2048,training_samples-start),cover.means.shape[1]))
            trained.extend(log_lr(y,cover.means,weights))
        threshold=float(np.partition(np.asarray(trained),calibration_order-1)[calibration_order-1])
        tie_probability=0.0  # strict rejection remains valid with atoms
    evaluated = mixture_scores(cover.means, weights, samples, generators[1])
    rejects = _reject(evaluated,threshold,tie_probability,generators[1])
    lam,w,h,slack,nominal_null_upper,gamma,dual = _importance_bounds(
        evaluated,rejects,threshold,alpha_work,alpha+null_tv,failure)
    if calibration=="null_order_statistic":
        nominal_null_upper=min(nominal_null_upper,alpha_work)
        gamma=1.0
    model_errors = np.minimum(1.0, cover.tv + alternative_tv)
    average_error = float(weights @ model_errors)
    # A true-null-feasible test has nominal size <= alpha + null_tv.
    upper = min(1.0, dual + average_error)
    if objective == "average":
        p_lower = _lower_binomial(int(np.sum(rejects)), samples, failure)
        lower = gamma * max(0.0, p_lower-average_error)
        component_lower = None
        component_count = 0
    else:
        component_count = component_samples or samples
        component_lower = []
        for i in range(len(cover.means)):
            scores = mixture_scores(cover.means, weights, component_count, generators[2], component=i)
            p = _lower_binomial(int(np.sum(_reject(scores,threshold,tie_probability,generators[2]))), component_count, failure)
            component_lower.append(gamma * max(0.0, p-model_errors[i]))
        lower = min(component_lower)
    # A data-independent coin is always feasible and has power alpha.
    # It can replace a poorer certified candidate without changing the upper bound.
    use_coin = lower < alpha
    lower = max(alpha, lower)
    gap = upper-lower
    # An inverted interval is a diagnostic failure; never hide it by sorting.
    if gap < -1e-12:
        raise ArithmeticError("Inconsistent certificate bounds; investigate or confidence event failed")
    return {
        "objective": objective, "alpha": alpha, "simultaneous_confidence": confidence,
        "optimal_power_interval": [float(lower), float(upper)],
        "certified_detector_power_lower": float(lower), "optimality_gap_upper": float(max(0,gap)),
        "requested_gap": requested_gap, "gap_target_met": bool(gap <= requested_gap),
        "detector": {"kind": "constant_coin" if use_coin else "randomized_mixture_lr",
                     "log_lr_threshold": threshold, "acceptance_probability_above_threshold": gamma,
                     "boundary_acceptance_probability": tie_probability,
                     "constant_coin_probability": alpha if use_coin else None,
                     "weights": weights.tolist()},
        "null_probability_upper": alpha if use_coin else min(1.0, gamma*nominal_null_upper+null_tv),
        "continuous_average_tv_error": average_error,
        "continuous_maximum_tv_error": float(np.max(model_errors)),
        "null_tv_radius": null_tv, "extra_alternative_tv_radius": alternative_tv,
        "dual_lambda": lam, "dual_h_mean": float(np.mean(h)),
        "threshold_calibration":calibration,"null_calibration_order":calibration_order,
        "importance_w_mean": float(np.mean(w)), "bounded_expectation_slack": slack,
        "component_power_lower_bounds": component_lower,
        "work": {"training_samples": training_samples, "mixture_validation_samples": samples,
                 "validation_samples_per_component": component_count,
                 "total_validation_samples": samples + component_count*len(cover.means),
                 "templates": len(cover.means), "score_dimension": cover.means.shape[1]},
        "arithmetic": "float64 score/statistical evaluation; not a formal roundoff certificate",
        "confidence_scope": "Joint over fresh simulation draws, conditional on valid model and cover",
    }


def rejection_probability(observation, means, report):
    """Executable certified decision rule; caller can draw a Bernoulli decision."""
    detector=report["detector"]
    if detector["kind"]=="constant_coin":
        return float(detector["constant_coin_probability"])
    score=float(log_lr(np.asarray(observation,float),means,detector["weights"]))
    threshold=detector["log_lr_threshold"]
    base=1.0 if score>threshold else detector["boundary_acceptance_probability"] if score==threshold else 0.0
    return base*detector["acceptance_probability_above_threshold"]


def flux_bracket(unit_flux_cover, *, maximum_flux, target_power=.9, alpha=.01,
                 confidence=.99, relative_tolerance=.02, max_evaluations=16,
                 samples=262144, max_samples=4194304, training_samples=131072,
                 seed=2026, calibration="alternative"):
    """Bracket the prior-average Gaussian flux limit, with an explicit gap.

    All source means and radii scale with flux; noise and cell prior are fixed.
    Power is nondecreasing by the Gaussian degradation channel. Allocate the
    failure probability across the maximum number of adaptive evaluations.
    An unresolved power interval never licenses discarding half the flux range.
    """
    if not np.isfinite(maximum_flux) or maximum_flux<=0:
        raise ValueError("Positive finite maximum flux required")
    if not 0<alpha<target_power<1 or not 0<confidence<1 or not 0<relative_tolerance<1:
        raise ValueError("Require 0 < alpha < target power < 1 and valid confidence/tolerance")
    if not isinstance(max_evaluations,int) or max_evaluations<1:
        raise ValueError("Positive maximum evaluation count required")
    if not isinstance(max_samples,int) or max_samples<samples:
        raise ValueError("Maximum sample budget must be at least samples")
    per_confidence=1-(1-confidence)/max_evaluations
    children=np.random.SeedSequence(seed).spawn(max_evaluations)
    lower,upper=0.,None
    evaluations=[]
    candidate=float(maximum_flux)
    budget=samples
    for i in range(max_evaluations):
        scaled=GaussianCover(unit_flux_cover.means*candidate,unit_flux_cover.radii*candidate,
                             unit_flux_cover.weights)
        report=certify(scaled,alpha=alpha,confidence=per_confidence,samples=budget,
                       training_samples=training_samples,seed=int(children[i].generate_state(1)[0]),
                       calibration=calibration)
        lo,hi=report["optimal_power_interval"]
        evaluations.append({"flux":candidate,"certificate":report})
        if hi<target_power:
            lower=max(lower,candidate)
        elif lo>=target_power:
            upper=candidate if upper is None else min(upper,candidate)
        else:
            if budget<max_samples:
                budget=min(max_samples,4*budget)
                continue
            break
        if upper is None:
            # The supplied upper search limit has not been shown sufficient.
            break
        if upper-lower<=relative_tolerance*upper:
            break
        candidate=(lower+upper)/2
        budget=samples
    width=None if upper is None else upper-lower
    return {"objective":"prior-average minimum flux attaining target power",
            "target_power":target_power,"alpha":alpha,"simultaneous_confidence":confidence,
            "flux_interval":[lower,upper],"absolute_width":width,
            "relative_width":None if upper is None else width/upper,
            "requested_relative_tolerance":relative_tolerance,
            "gap_target_met":bool(upper is not None and width<=relative_tolerance*upper),
            "evaluations":evaluations,"maximum_evaluations":max_evaluations,
            "uncertainty":"Conditional Monte Carlo confidence plus supplied continuous cover; float64 evaluation",
            "status":"bracketed" if upper is not None else "upper flux not shown sufficient"}


def least_favorable_weights(means, *, alpha=0.01, samples=32768, seed=12026):
    """Numerically minimize the dual; independent certify() supplies the bound.

    Fixed importance proposal R=(P0+uniform-mixture)/2 makes p_i/R <= 2K
    and p0/R <= 2. Optimizer success is never used as a proof of optimality.
    """
    means = np.asarray(means, float)
    if means.ndim != 2 or min(means.shape) < 1 or not np.all(np.isfinite(means)):
        raise ValueError("Finite nonempty means required")
    if not 0 < alpha < 1 or not isinstance(samples, int) or samples < 1:
        raise ValueError("Valid alpha and sample count required")
    k, dimension = means.shape
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(samples, dimension))
    labels = rng.integers(0, k, samples)
    y += means[labels] * (rng.random(samples) < 0.5)[:, None]
    logs = y @ means.T - np.sum(means**2, axis=1)/2
    proposal = np.logaddexp(0.0, logsumexp(logs, axis=1)-math.log(k))-math.log(2)
    a, b = np.exp(logs-proposal[:, None]), np.exp(-proposal)
    def objective(x):
        residual = a @ x[:k] - x[-1]*b
        active = residual > 0
        value = x[-1]*alpha + np.maximum(residual, 0).mean()
        gradient = np.r_[a[active].sum(axis=0)/samples, alpha-b[active].sum()/samples]
        return value, gradient
    fit = minimize(objective, np.r_[np.full(k, 1/k), 1.0], jac=True, method="SLSQP",
                   bounds=[(0,1)]*k+[(0,1/alpha)],
                   constraints=[{"type":"eq", "fun":lambda x: x[:k].sum()-1,
                                 "jac":lambda x: np.r_[np.ones(k), 0.0]}],
                   options={"maxiter":300, "ftol":1e-10})
    return _weights(np.maximum(0, fit.x[:k]), k), {
        "optimizer_success": bool(fit.success), "message": str(fit.message),
        "training_objective": float(fit.fun), "training_samples": samples,
        "iterations": int(fit.nit), "certificate": False,
    }


def geometric_detector(cover, alpha=0.01):
    """Convex-hull separating direction; only its lower bound is constructive.

    Convexifying the signal means does NOT give a valid upper bound for the
    original nonconvex family. An oracle-component upper bound is used instead.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    k = len(cover.means)
    gram = cover.means @ cover.means.T
    fit = minimize(lambda w: (float(w @ gram @ w), 2*gram@w), np.full(k,1/k), jac=True,
                   bounds=[(0,1)]*k, method="SLSQP",
                   constraints=[{"type":"eq", "fun":lambda w:w.sum()-1,
                                 "jac":lambda w:np.ones(k)}],
                   options={"ftol":1e-13,"maxiter":500})
    direction = _weights(np.maximum(0,fit.x), k) @ cover.means
    distance = float(np.linalg.norm(direction))
    if distance < 1e-12:
        lower = alpha
        direction = np.zeros(cover.means.shape[1])
    else:
        direction /= distance
        lower = max(alpha, float(np.min(norm.cdf(cover.means @ direction-cover.radii-norm.isf(alpha)))))
    upper = min(1.0, float(np.min(norm.cdf(np.linalg.norm(cover.means,axis=1)-norm.isf(alpha))+cover.tv)))
    return {"objective":"minimax", "power_interval":[lower,upper],
            "gap":upper-lower, "direction":direction.tolist(),
            "convex_hull_distance":distance, "optimizer_success":bool(fit.success),
            "arithmetic":"float64; analytic model-conditional bounds"}
