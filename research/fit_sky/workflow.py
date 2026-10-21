"""Shared sequential FitSky workflow for the direct and agent interfaces."""
import json

from .sequential import SequentialFit


def prepare(value, settings, *, resume=None, stop_after=None, exposure_batch=1):
    if exposure_batch < 1:
        raise ValueError("exposure-batch must be positive")
    if resume:
        fit = SequentialFit.load(resume, value["scene"])
        if any(fit.settings[k] != v for k, v in settings.items()):
            raise ValueError("Resume settings conflict with checkpoint; omit overrides or start a new fit")
        if "queries" in value:
            fit.model.events(fit.population, value["queries"])
            fit.queries = value["queries"]
    else:
        fit = SequentialFit(value["scene"], queries=value.get("queries"), **settings)
    stop = len(value["scene"]["exposures"]) if stop_after is None else stop_after
    if not fit.assimilated < stop <= len(value["scene"]["exposures"]):
        raise ValueError("stop-after must append at least one available exposure")
    return fit, stop


def advance(fit, directory, *, stop, exposure_batch=1, catalogues=False, on_progress=None):
    # Save the starting state too, so interruption in the first batch is resumable.
    fit.save(directory / "checkpoint.npz")
    while fit.assimilated < stop:
        fit.assimilate(min(stop, fit.assimilated + exposure_batch))
        fit.save(directory / "checkpoint.npz")
        report = fit.report(include_catalogues=catalogues)
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "report.json")
        if on_progress:
            on_progress(fit.summary())
    return report, fit.summary()
