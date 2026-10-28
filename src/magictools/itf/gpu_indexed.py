"""Reuse fixed observation vectors on CUDA; identical propagation arithmetic.

Only array addressing differs. A generated kernel reuses the exact existing
Python body so light-time, derivatives, convergence and invalidity rules cannot
silently diverge between copied and indexed inputs.
"""
import ast
import inspect
import textwrap
import time
import numpy as np
from numba import cuda
from .gpu import _evaluate

_NAMES = ('observers','observed','east','north')


def indexed_body():
    function = ast.parse(textwrap.dedent(inspect.getsource(_evaluate.py_func))).body[0]
    function.name = '_evaluate_indexed'
    function.decorator_list = []
    function.args.args.insert(7,ast.arg(arg='observation_index'))
    class Address(ast.NodeTransformer):
        count = 0
        def visit_Subscript(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id in _NAMES:
                if not isinstance(node.slice,ast.Tuple) or not isinstance(node.slice.elts[0],ast.Name) or node.slice.elts[0].id!='i':
                    raise ValueError('Unexpected geometry access in reference kernel')
                node.slice.elts[0]=ast.Subscript(value=ast.Name(id='observation_index',ctx=ast.Load()),slice=ast.Name(id='i',ctx=ast.Load()),ctx=ast.Load())
                self.count += 1
            return node
    transform=Address();function=transform.visit(function)
    if transform.count != 18:
        raise ValueError('Reference kernel geometry access changed; review indexed transform')
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(_evaluate.py_func.__globals__)
    exec(compile(module,__file__,'exec'),namespace)
    return namespace['_evaluate_indexed']


_evaluate_indexed=cuda.jit(fastmath=False)(indexed_body())


class ResidentGeometry:
    def __init__(self, arrays, *, allow_simulator=False):
        if cuda.config.ENABLE_CUDASIM and not allow_simulator:
            raise RuntimeError('CUDA simulator is not a GPU benchmark')
        if not cuda.is_available(): raise RuntimeError('No CUDA device')
        started=time.perf_counter();self.arrays={}
        for name in _NAMES:
            value=np.asarray(arrays[name])
            if value.dtype!=np.float64 or value.ndim!=2 or value.shape[1]!=3 or not np.isfinite(value).all():
                raise ValueError('Resident geometry requires finite float64 (N,3) arrays')
            self.arrays[name]=np.array(value,copy=True,order='C')
            self.arrays[name].flags.writeable=False
        if len({x.shape for x in self.arrays.values()})!=1:raise ValueError('Resident geometry shapes differ')
        self.inputs=[cuda.to_device(self.arrays[name]) for name in _NAMES]
        cuda.synchronize()
        self.setup_seconds=time.perf_counter()-started
        self.upload_bytes=sum(a.nbytes for a in self.arrays.values())
        self.bindings=0;self.avoided_upload_bytes=0;self.validation_seconds=0.

    @classmethod
    def from_atlas(cls,geometry,**kwargs):
        if not hasattr(geometry,'observation_indices'):raise ValueError('Atlas lacks exact observation indices')
        start=time.perf_counter();u=np.asarray(geometry.unit)
        ra=np.arctan2(u[:,1],u[:,0]);east=np.column_stack((-np.sin(ra),np.cos(ra),np.zeros(len(u))))
        value=cls(dict(observers=np.asarray(geometry.observers),observed=u,east=east,north=np.cross(u,east)),**kwargs)
        value.total_preparation_seconds=time.perf_counter()-start
        return value

    def bind(self,bank,arrays):
        start=time.perf_counter();index=np.asarray(bank.get('observation_index'))
        if index.dtype!=np.int64 or index.shape!=(len(arrays['offsets']),) or not index.flags.c_contiguous:
            raise ValueError('Missing or invalid exact observation indices')
        if not len(index) or index.min()<0 or index.max()>=len(self.arrays['observed']):
            raise ValueError('Resident observation index outside catalogue')
        # No covariance/ephemeris substitution: verify every addressed vector
        # against the actual CPU fitting bank, including signed-zero bits.
        for name in _NAMES:
            if not np.array_equal(self.arrays[name][index].view(np.uint64),arrays[name].view(np.uint64)):
                raise ValueError('Resident '+name+' differs from fitted observation bank')
        self.validation_seconds+=time.perf_counter()-start
        self.bindings+=1
        self.avoided_upload_bytes+=sum(arrays[k].nbytes for k in _NAMES)-index.nbytes
        return self.inputs+[cuda.to_device(index)]
