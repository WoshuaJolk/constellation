"""Execute the stage-aware hierarchy, retaining whole-box error accounting."""
from time import perf_counter
import numpy as np
from flint import arb

from research.detection_limit.interval_orbits import upper_float
from ..cascade_domain import certify_domain
from ..cascade_transfer import CascadeTransfer
from ..local_hierarchy import LocalHierarchyPlan, LocalHierarchy, PlannedNode, transfer_cost, _down
from ..segment_field import TensorGrid
from .resolution import local_grid, PERMUTATION


class OrderedHierarchyPlan(LocalHierarchyPlan):
    def _grid(self,proof,time_range,tolerance):
        # The baseline restart cache is deliberately not used for this different
        # stage certificate. Whole completed plans can still be saved/replayed.
        return local_grid(proof,self.observer,time_range,self.frequency_bounds,tolerance,**self.options)

    def _node(self,first,last,epoch,center,widths,grid,budget):
        node=PlannedNode(first,last,epoch,center,widths,grid,budget)
        self.nodes.append(node)
        if self.progress is not None:
            self.progress({'first':first,'last':last,'grid_nodes':grid['nodes'],
                           'cells':grid['cells'],'orders':grid['orders']})
        if last-first==1:
            return node
        children,edges,errors=[],[],[]
        cut=(first+last)//2
        for a,b in ((first,cut),(cut,last)):
            child_epoch=float((self.times[a]+self.times[b-1])/2)
            proof=certify_domain(self.observer,center,widths,epoch,child_epoch,**self.certificate_options)
            child_grid=self._grid(proof,[self.times[a],self.times[b-1]],_down(arb(budget)/2))
            child_budget=_down(arb(budget)/(2*arb(child_grid['input_error_amplification'])))
            child=self._node(a,b,child_epoch,proof['child_center'],proof['child_widths'],child_grid,child_budget)
            cost=transfer_cost(grid['shape'],child_grid['shape'],child_grid['orders'],permutation=PERMUTATION)
            edge={'first':a,'last':b,'certificate':proof,'cost':cost,'permutation':list(PERMUTATION)}
            self.edges.append(edge)
            errors.append(arb(child_grid['error_bound'])+arb(child_grid['input_error_amplification'])*child.error)
            children.append(child);edges.append(edge)
        node.children,node.edges=tuple(children),tuple(edges)
        node.error=upper_float(max(errors))
        if not arb(node.error)<=arb(budget):
            raise ValueError('Stage-aware subtree exceeded its error budget')
        return node

    def materialize(self,frequencies,*,max_nodes=1_000_000,max_entries=20_000_000):
        if type(max_nodes) is not int or type(max_entries) is not int or min(max_nodes,max_entries)<1:
            raise ValueError('Positive integer allocation budgets required')
        if (self.stats['total_stored_field_nodes']>max_nodes
                or self.stats['peak_field_or_intermediate_nodes']>max_nodes
                or self.stats['total_transfer_entries']>max_entries):
            raise MemoryError('Sufficient ordered hierarchy exceeds allocation budgets; plan retained')
        return OrderedHierarchy(self,frequencies,max_entries=max_entries)


class OrderedHierarchy(LocalHierarchy):
    def __init__(self,plan,frequencies,*,max_entries):
        self.plan,self.max_entries=plan,max_entries
        self.frequencies=[np.asarray(k,float) for k in frequencies]
        K=np.asarray(plan.frequency_bounds)
        if (len(self.frequencies)!=len(plan.times) or any(
                k.ndim!=2 or k.shape[1]!=2 or not len(k) or not np.isfinite(k).all()
                or np.any(abs(k)>K) for k in self.frequencies)):
            raise ValueError('Exposure frequencies must lie inside the planned bounds')
        leaf_entries=sum(n.grid['nodes']*len(self.frequencies[n.first]) for n in plan.nodes if not n.children)
        if leaf_entries+plan.stats['total_transfer_entries']>max_entries:
            raise MemoryError('Combined leaf and transfer budget exceeded before allocation')
        self.offsets=np.r_[0,np.cumsum([len(k) for k in self.frequencies])]
        self.grids,self.transfers,self.kernels={},{},{}
        begin=perf_counter()
        for node in plan.nodes:
            grid=TensorGrid(node.center,node.widths,tuple(node.grid['cells']),tuple(node.grid['orders']))
            self.grids[id(node)]=grid
            if not node.children:
                self.kernels[id(node)]=np.exp(1j*grid.points()[:,[3,4]] @ self.frequencies[node.first].T)
        for node in plan.nodes:
            for child,edge in zip(node.children,node.edges):
                proof=edge['certificate']
                chart,target=plan.observer.chart(node.epoch),plan.observer.chart(child.epoch)
                def rebase(q,chart=chart,target=target,node=node,child=child,proof=proof):
                    if np.any(abs((q-np.asarray(node.center))/node.widths)>np.array(proof['normalized_enclosing_radii'])):
                        raise ValueError('Numerical geometry left its certified domain')
                    return chart.rebase(q,child.epoch-node.epoch,target)
                self.transfers[id(child)]=CascadeTransfer(self.grids[id(node)],self.grids[id(child)],
                                                          rebase,max_entries=max_entries,permutation=PERMUTATION)
        self.stats={**plan.stats,'grid_allocated':True,'materialization_seconds':perf_counter()-begin,
                    'actual_transfer_entries':sum(t.entries for t in self.transfers.values()),
                    'actual_geometry_state_evaluations':sum(t.geometry_points for t in self.transfers.values()),
                    'leaf_kernel_entries':sum(k.size for k in self.kernels.values()),
                    'permutation':list(PERMUTATION)}
