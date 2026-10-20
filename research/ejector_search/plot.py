"""Standalone summary of bounded decisions and the blinded source recovery."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    reports = [json.loads((HERE/'results'/f'{case}-bounds/complete.json').read_text()) for case in ['observed', 'injection']]
    recovery = json.loads((HERE/'results/recovery.json').read_text())
    sources = recovery['candidate']['counterparts']['nearest']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.set_facecolor('#f9fafb')
    colors = ['#334155', '#10b981']
    for i, (r, label) in enumerate(zip(reports, ['18 observed stars', '21 stars: real + injection'])):
        axes[0].barh(i, r['excluded_count'], color=colors[0], height=.48)
        axes[0].barh(i, r['certified_compatible_count'], left=r['excluded_count'], color=colors[1], height=.48)
        axes[0].text(10, i, f"{r['excluded_count']:,} excluded", color='white', va='center', fontsize=11)
        axes[0].text(r['hypotheses']+25, i, f"{r['certified_compatible_count']} verified\nencounter", va='center', fontsize=10)
    axes[0].set_yticks([0, 1], ['18 observed stars', '21 stars: real + injection'])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1570)
    axes[0].set_xlabel('All triple hypotheses\nZero unresolved in both runs')
    axes[0].set_title('Every group decided\nwithin the stated domain', loc='left', fontsize=12)
    names = [s['id'].replace('_', ' ') for s in sources]
    distances = [s['separation_pc'] for s in sources]
    axes[1].barh(names, distances, color=['#10b981']+['#94a3b8']*4, height=.6)
    axes[1].invert_yaxis()
    for i, d in enumerate(distances): axes[1].text(d+.6, i, f'{d:.2f}', va='center', fontsize=10)
    axes[1].set_xlim(0, max(distances)+12)
    axes[1].set_xlabel('Nominal cluster–origin separation\nat encounter (pc)')
    axes[1].set_title('Recovered injection source\n5,749 supplied clusters', loc='left', fontsize=12)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)
    fig.suptitle('Continuous-time encounter certificates and source recovery', x=.04, ha='left', fontsize=15)
    fig.text(.04, .025, 'Domain: total measurement cost ≤ 25, radius 1 pc, age 0.1–6 Myr, fixed Galactic model.\nModel-conditional decisions; injected stars are synthetic. No new astronomical discovery or all-survey completeness.', fontsize=9, color='#475569')
    fig.tight_layout(rect=[0, .12, 1, .94], w_pad=2.5)
    fig.savefig(HERE/'results/certified-search.png', dpi=180, facecolor=fig.get_facecolor())


if __name__ == '__main__': main()
