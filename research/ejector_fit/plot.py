#!/usr/bin/env python3
"""Make the standalone scientific summary plot (requires matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    def load(name): return json.loads((args.campaign/'fits'/f'{name}.json').read_text())['best_fit']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    names = ['hip1997_triple', 'hip2007_triple', 'gaia_triple']
    costs = [load(name)['chi2'] for name in names]
    labels = ['Historical Hipparcos', 'Revised Hipparcos', 'Gaia DR3*']
    bars = axes[0].barh(labels, costs, color=['#337e9a', '#b76748', '#b76748'])
    axes[0].bar_label(bars, fmt='%.2f', padding=4)
    axes[0].set_xlim(0, 44)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Squared Mahalanobis measurement shift, χ²\nFit cost; not a discovery probability')
    axes[0].set_title('Same exact Orion triple, different observations')
    axes[0].text(0, -.24, '*ι Ori has unreliable Gaia single-star astrometry (RUWE 8.2).', transform=axes[0].transAxes, fontsize=8)
    fit = load('Orion_network')
    events = fit['events']
    ages = [e['lookback_myr'] for e in events]
    labels = ['AE Aur — ONC (≤5 pc)', 'μ Col — ONC (≤5 pc)', 'μ Col — Collinder 69 (≤7 pc)']
    axes[1].scatter(ages, np.arange(3), s=90, color=['#337e9a', '#7359a2', '#7359a2'])
    axes[1].plot(ages[1:], [1, 2], color='#7359a2', alpha=.5)
    for j, age in enumerate(ages): axes[1].annotate(f'{age:.3f} Myr', (age, j), xytext=(6, 8), textcoords='offset points', fontsize=9)
    axes[1].set_yticks(np.arange(3), labels)
    axes[1].set_xlim(2.25, 3.65)
    axes[1].set_ylim(-.5, 2.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Lookback time (Myr)\nFitted points, not uncertainty intervals')
    axes[1].set_title('One shared trajectory can pass\ntwo proposed source clusters')
    axes[1].text(0, -.24, f'Joint χ² = {fit["chi2"]:.3f}; largest measurement shift <1.49σ.\nPassage compatibility does not identify the birthplace.', transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Observed-star validation of ejector_fit — no new ejection discovery', fontsize=13)
    fig.savefig(args.campaign/'encounters.png', dpi=180)


if __name__ == '__main__': main()
