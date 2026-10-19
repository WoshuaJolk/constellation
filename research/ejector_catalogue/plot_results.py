#!/usr/bin/env python3
"""Optional static campaign figure (requires matplotlib)."""
import gzip
import json
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from magictools.ejector_gaia import read_gaia

HERE=Path(__file__).resolve().parent
raw=read_gaia(HERE/'data/gaia_fast.fits.gz')
with gzip.open(HERE/'results/final-baseline.json.gz','rt') as f:baseline=json.load(f)
with gzip.open(HERE/'results/baseline-input-audit.json.gz','rt') as f:audit=json.load(f)
measured={r['source_id'] for r in audit['ledger'] if r.get('radial_velocity')=='measurement'}
mask=np.array([str(x) in measured for x in raw['source_id']])
sky=SkyCoord(ra=np.array(raw['ra'])*u.deg,dec=np.array(raw['dec'])*u.deg).galactic
longitude=np.deg2rad((sky.l.deg+180)%360-180);latitude=sky.b.rad
fig=plt.figure(figsize=(12,4.8),facecolor='white')
ax=fig.add_subplot(121,projection='mollweide')
ax.scatter(longitude[~mask],latitude[~mask],s=.6,color='#94a3b8',alpha=.35,rasterized=True,label='RV requires a prior (25,572)')
ax.scatter(longitude[mask],latitude[mask],s=.7,color='#2563eb',alpha=.45,rasterized=True,label='RV passes initial cuts (13,317)')
ax.grid(alpha=.3);ax.tick_params(labelsize=7);ax.set_title('38,889 actual Gaia DR3 sources',fontsize=12,pad=16)
ax.set_xlabel('Galactic longitude',fontsize=9);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.24),fontsize=8,markerscale=4,frameon=False)
ax=fig.add_subplot(122)
rows=baseline['candidates'];ax.scatter([r['lookback_myr'] for r in rows],[r['support'] for r in rows],s=24,
    color='#64748b',alpha=.7,label='Real-data groups at 10 pc')
injection=json.loads((HERE/'results/injection-1pc.json').read_text())['candidates'][0]
ax.scatter([injection['lookback_myr']],[injection['support']],s=150,marker='*',color='#0f766e',edgecolors='white',linewidths=.5,label='Injected event at 1 pc',zorder=5)
ax.set(xlim=(-.1,10),ylim=(0,1.08),xlabel='Lookback time (Myr)',ylabel='Fraction of 16 joint draws',title='Compact-event recovery and the real-data null')
ax.text(.97,.48,'No real group passes the 1 pc test\nin the baseline sampled search.',transform=ax.transAxes,ha='right',fontsize=9,color='#334155')
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.suptitle('find_ejector: catalogue search, not a confirmed discovery',fontsize=14,y=.99)
fig.subplots_adjust(left=.05,right=.98,bottom=.23,top=.85,wspace=.3)
fig.text(.5,.025,'Conditional on the declared selection, potential, priors, 10 Myr interval and sampled support threshold.',ha='center',fontsize=8,color='#475569')
fig.savefig(HERE/'results/campaign.png',dpi=170)
