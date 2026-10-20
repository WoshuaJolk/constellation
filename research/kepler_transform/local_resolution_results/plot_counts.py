"""Plot verified size estimates; this is not a timing or speedup chart."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).parent
previous=json.loads((root.parent/'constructive_results/local.json').read_text())['certificates']
cases=['near','distant','stress']
uniform=[x['sufficient_single_cell_order_not_allocated']['uniform_single_cell_grid_nodes'] for x in previous]
local=[json.loads((root/f'local-{case}/results.json').read_text())['local']['grid']['nodes'] for case in cases]
ordered=[json.loads((root/f'ordered/local-{case}/results.json').read_text())['local']['grid']['nodes'] for case in cases]
base=json.loads((root/'hierarchy-near/results.json').read_text())['hierarchy']['atlas']['stats']
better=json.loads((root/'ordered/hierarchy-near/results.json').read_text())['hierarchy']['plan']['stats']
fig,axes=plt.subplots(1,2,figsize=(11.5,5.4))
fig.subplots_adjust(top=.76,bottom=.22,left=.08,right=.98,wspace=.3)
x=np.arange(3)
colors=['#8b96a7','#386b99','#197c80']
labels=['Previous uniform order','Local cells and orders','Stage-aware local grids']
handles=[]
for offset,values,color,label in zip([-.25,0,.25],[uniform,local,ordered],colors,labels):
 handles.append(axes[0].bar(x+offset,values,.24,label=label,color=color))
axes[0].set(yscale='log',ylim=(100,1e15),xticks=x,xticklabels=['Near','Distant','Stress'],
            ylabel='Sufficient child input nodes (log scale)',title='Same 0.125-day transfer and original domains')
for i,n in enumerate(ordered):
 axes[0].text(i+.25,n*1.7,str(n),ha='center',va='bottom',fontsize=8,color='#12585b')
x=np.arange(2)
for offset,stats,color in [(-.18,base,colors[1]),(.18,better,colors[2])]:
 counts=[stats['total_stored_field_nodes'],stats['total_transfer_entries']]
 axes[1].bar(x+offset,counts,.35,color=color)
 for i,n in enumerate(counts):
  axes[1].text(i+offset,n*1.35,f'{n:.3g}',ha='center',fontsize=8)
axes[1].set(yscale='log',ylim=(1e9,2e15),xticks=x,xticklabels=['All stored\nfield values','All interpolation\nweights'],
            ylabel='Number of entries (log scale)',title='Complete eight-day near-domain tree')
for axis in axes:
 axis.spines[['top','right']].set_visible(False)
 axis.grid(axis='y',which='major',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('Local grids and a different coordinate order reduce the verified counts',fontsize=13,y=.97)
fig.legend([h[0] for h in handles],labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.05,'Error target: 1e-8 per unit coefficient l1 norm. Large graphs are certified plans, not allocated archive runs.',ha='center',fontsize=9)
fig.savefig(root/'size_comparison.svg')
fig.savefig(root/'size_comparison.png',dpi=150)
