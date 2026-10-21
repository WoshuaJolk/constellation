"""Release promotion records: bounded usefulness, not a claim of complete science."""
READY = {
    'find_frames': 'Federated frame discovery, deduplication and explicit service failures',
    'check_frames': 'Coverage-evidence review and response-integrity checks',
    'kepler-propagate': 'Two-body propagation of supplied states with optional light time',
    'map_cover': 'Usable local-image coverage including masks and off-chip predictions',
    'force_track': 'Signed forced photometry with flux uncertainties',
    'dig_floor': 'Joint scoring of supplied finite trajectories',
    'inject_tracks': 'Pixel-injection recovery of the actual search procedure',
    'judge_line': 'Candidate consistency checks and control tracks',
    'weigh_stamp': 'Conditional pixel-model comparisons for candidate diagnostics',
    'learn_null': 'Discrete hypothesis updates and information-based follow-up',
    'map_complete': 'Exact finite selection probabilities with shared exposures',
    'inject_all': 'Finite selection Monte Carlo beyond exact enumeration limits',
    'match_all': 'Optimal or bounded assignment over supplied scored candidates',
    'verify_match': 'Independent verification of assignment certificates',
    'prove_floor': 'Prior-average detection-power bounds for a specified Gaussian family',
    'ejector_fit': 'Measured-astrometry fits of specified shared stellar-origin hypotheses',
}
EXPERIMENTAL = {
    'solve_itf': 'Small-scene finite-prior linkage reference; full-ITF continuous-domain search and 180-second target remain open.',
    'fold_kepler': 'Repeated-bank speedups exist, but per-invocation compilation can make the CLI slower than direct evaluation; no general speedup claim.',
    'fold-chart': 'Research transform with bounded geometry and unresolved general numerical/scaling guarantees.',
    'fit_sky': 'Experimental joint inference; approximate mode coverage and broad-scene accuracy remain unresolved.',
    'find_ejector': 'Specialist traceback search requires justified posterior inputs and calibrated population controls; not promoted by the release acceptance suite.',
    'ejector_prepare': 'Specialist Gaia preparation depends on supplied distance/RV priors; not promoted by the release acceptance suite.',
    'ejector_search': 'Specialist group search has bounded validation but lacks a general-purpose release acceptance case.',
    'ejector_bound': 'Specialist interval certificates depend on fixed-model domains and optional arithmetic dependencies; not promoted by this release suite.',
    'ztf_search': 'Legacy ZTF-specific discovery contract; use find_frames for the supported general archive workflow.',
    'ztf_fetch': 'Legacy ZTF preparation workflow uses an estimated local noise model; generic validated frame-to-pixel ingestion remains open.',
}


def catalogue():
    return {'ready': READY, 'experimental': EXPERIMENTAL,
            'evidence': 'docs/release-readiness.md; tests/test_release_cli.py',
            'meaning': 'Ready means a CLI acceptance case demonstrates a concrete improvement under its stated assumptions, not global scientific completeness.'}
