"""Replay the retained adaptive cover and run its fixed validation budget."""

import hashlib
import json
from pathlib import Path
import shutil
from flint import ctx

from research.fit_sky.experiments import save
from research.fit_sky.statistical import infer
from research.fit_sky.scene import Scene
from research.fit_sky.reference_quadrature import integrate
from research.fit_sky.arithmetic import lower, upper
from flint import arb


base = Path(__file__).resolve().parent
output = base/"validation"
output.mkdir(exist_ok=False)
snapshot = output/"source_snapshot"
snapshot.mkdir()
for source in base.parent.glob("*.py"):
    shutil.copy2(source, snapshot/source.name)
contract = json.loads((base/"validation-contract.json").read_text())
cover = json.loads((base/"joint-cover.json").read_text())
with ctx.workprec(100):
    print("Replay and fixed-budget integration", flush=True)
    result = infer(cover["manifest"], cover_report=cover, samples=contract["samples"],
                   seed=contract["seed"], confidence=contract["confidence"], marginalize_flux=True)
    result["manifest"] = cover["manifest"]
    save(output/"joint.json", result)
    print(result["status"], result["relative_evidence_gap"], result["queries"], flush=True)
    references = []
    for order in (20, 32):
        references.append(integrate(cover["manifest"], order=order))
        save(output/"quadrature.json", references)
    # The preceding training run used 99.5% confidence, preserving >=99%
    # simultaneous coverage for this ratio by the union bound.
    training = json.loads((base.parent/"noise_floor_predictive_results/training.json").read_text())
    tscene, jscene = Scene(training["manifest"]), Scene(cover["manifest"])
    tlo, thi = training["scaled_evidence"]
    jlo, jhi = result["scaled_evidence"]
    shift = jscene.log_normalization-tscene.log_normalization
    bounds = ([lower(arb(jlo).log()-arb(thi).log()+shift),
               upper(arb(jhi).log()-arb(tlo).log()+shift)] if min(tlo, jlo) > 0 else [None, None])
    save(output/"predictive.json", {
        "log_predictive_density_bounds": bounds, "simultaneous_confidence_lower": .99,
        "conditional_on_stated_model_and_float_sampling_assumptions": True,
        "heldout_used_to_choose_prior": False, "full_objective_complete": False})
    print("Predictive", bounds, flush=True)
save(output/"inventory.json", {
    str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(output.rglob("*")) if p.is_file()})
