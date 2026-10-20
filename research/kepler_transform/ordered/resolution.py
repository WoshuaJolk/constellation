"""Preserve child-span sensitivities by interpolating hidden coordinates first.

The analytic field at a stage is expressed in child coordinates. Unconverted
child coordinates are EXACT prescribed hybrid variables; their derivatives
must not be replaced by an all-components inverse majorant. This preserves
constant directions of instantaneous exposure fields.
"""
from time import perf_counter
import numpy as np
from flint import arb, arb_mat, acb

from research.detection_limit.interval_orbits import upper_float
from ..cascade_domain import _state_box
from ..local_resolution import (
    AXES, PERMUTATION as TENSOR_ERROR_ORDER, _time_envelope, _partition, _lebesgue, choose_axis, axis_error, grid_plan,
    axis_neighborhood,
)

PERMUTATION=(0,1,2,5,3,4)


def stage_phase(proof, observer, axis, ellipse, time_range, frequency_bounds,
                *, max_step=.03125, max_range_pieces=16):
    """Certify the field BEFORE replacing axis in the specified cascade order.

    Native stages require only holomorphy of the child field. General stages
    retain the old whole-domain inverse certificate, but exploit the exact
    coordinate identities when composing its derivative with the child flow.
    """
    begin=perf_counter()
    selected=PERMUTATION[:PERMUTATION.index(axis)]
    lo,hi=map(float,time_range)
    K=np.asarray(frequency_bounds,float)
    if (lo>hi or not np.isfinite([lo,hi]).all() or K.shape!=(2,)
            or not np.isfinite(K).all() or np.any(K<0) or ellipse<=1):
        raise ValueError('Valid interval, frequency rectangle and ellipse required')
    instant=lo==hi==proof['stop']
    constant=instant and axis in (0,1,2,5)
    native=(not selected or proof['start']==proof['stop']
            or (instant and not any(j in (3,4) for j in selected)))
    w=list(map(arb,proof['parent_widths']))
    cw=list(map(arb,proof['child_widths']))
    b=cw[axis]/w[axis]
    r=arb(float(ellipse)); a,s=(r+1/r)/2,(r-1/r)/2
    if constant:
        return {'axis':axis,'ellipse':float(ellipse),'hybrid_halfwidth':upper_float(b),
                'phase_derivative_bound':0.,'kind':'constant_instantaneous_stage',
                'selected_parent_axes':list(selected),'time_range':[lo,hi],
                'frequency_bounds':K.tolist(),'seconds':perf_counter()-begin,
                'constant_stage_certified':True}
    proof_inverse=None
    gamma=[arb(int(j==axis)) for j in AXES]
    real=list(cw)
    imag=[arb(0) for _ in AXES]
    real[axis]=cw[axis]*a
    imag[axis]=cw[axis]*s
    if not native:
        proof_inverse=axis_neighborhood(proof,observer,axis,ellipse,max_step=max_step)
        E=arb_mat(proof_inverse['increment_majorant'])
        S=arb_mat(proof_inverse['inverse_majorant'])
        I=arb_mat([[int(i==j) for j in AXES] for i in AXES])
        J=(I+E)*S
        # For a selected child coordinate, start at the real hybrid with the
        # current input coordinate clipped to its real box. Its real child
        # image lies in cw. Integrate the holomorphic derivative over the real
        # overhang, then the imaginary segment, inside the certified cylinder.
        # All unselected child coordinates keep their prescribed values.
        for j in selected:
            gamma[j]=J[j,axis].upper()
            real[j]=cw[j]+w[j]*gamma[j]*b*(a-1+s)
            imag[j]=w[j]*gamma[j]*b*s
    values=[acb(arb(float(c),rr.upper()),arb(0,ii.upper()))
            for c,rr,ii in zip(proof['child_center'],real,imag)]
    work={'attempted_steps':0,'refused_partitions':0}
    pieces=1
    while True:
        limits=[]
        try:
            for part in range(pieces):
                for end in sorted(set((min(lo,proof['stop']),max(hi,proof['stop'])))):
                    J=_time_envelope(_partition(values,pieces,part),proof['stop'],end,
                                     observer,proof['parent_widths'],work,max_step)
                    limits.append(sum(arb(float(k))*w[j]*sum(J[j,l]*gamma[l] for l in AXES)
                                      for k,j in zip(K,(3,4))).upper())
            break
        except ValueError:
            work['refused_partitions']+=1
            if pieces*2>max_range_pieces:
                raise
            pieces*=2
    phase=upper_float(max(limits))
    if not np.isfinite(phase):
        raise ValueError('Nonfinite stage phase majorant')
    return {'axis':axis,'ellipse':float(ellipse),'hybrid_halfwidth':upper_float(b),
            'phase_derivative_bound':phase,'kind':'native_child_field' if native else 'mixed_child_field',
            'selected_parent_axes':list(selected),'time_range':[lo,hi],'frequency_bounds':K.tolist(),
            'child_real_halfwidths':[upper_float(x) for x in real],
            'child_imaginary_halfwidths':[upper_float(x) for x in imag],
            'child_coordinate_derivative_majorant':[upper_float(x) for x in gamma],
            'inverse_neighborhood':proof_inverse,'work':work,'range_pieces':pieces,
            'all_times_in_interval_covered':True,'seconds':perf_counter()-begin}


def local_grid(proof,observer,time_range,frequency_bounds,tolerance=1e-8,*,
               neighborhood_ellipses=(1.05,2.,4.,8.,32.),max_order=64,max_step=.03125):
    begin=perf_counter()
    # Native tensor interpolation commutes: its error decomposition need not
    # use the physical cascade order. Keep the baseline tensor ordering there.
    error_order=TENSOR_ERROR_ORDER if proof["start"]==proof["stop"] else PERMUTATION
    if not np.isfinite(tolerance) or tolerance<=0 or not 1<=max_order<=64:
        raise ValueError('Positive tolerance and valid order budget required')
    target=float(np.nextafter(float((arb(tolerance)/(6*_lebesgue(max_order)**5)).lower()),0.))
    if target<=0:
        raise ValueError('Stage error budget underflows')
    choices,certificates,trials,options=[],[],[],[]
    for axis in AXES:
        best,winner,variants=None,None,[]
        # Exact constants need no complex inverse nor gravity evaluation.
        instant=time_range[0]==time_range[1]==proof['stop']
        radii=(2.,) if instant and axis in (0,1,2,5) else neighborhood_ellipses
        for ellipse in radii:
            started=perf_counter()
            try:
                certificate=stage_phase(proof,observer,axis,ellipse,time_range,frequency_bounds,max_step=max_step)
                variants.append(certificate)
                candidate=choose_axis(certificate,certificate['phase_derivative_bound'],target,max_order=max_order)
                trials.append({'axis':axis,'ellipse':ellipse,'status':'certified',
                               'kind':certificate['kind'],'axis_nodes':candidate['axis_nodes'],
                               'seconds':perf_counter()-started})
                if best is None or candidate['axis_nodes']<best['axis_nodes']:
                    best,winner=candidate,certificate
            except ValueError as error:
                trials.append({'axis':axis,'ellipse':ellipse,'status':'refused',
                               'reason':str(error),'seconds':perf_counter()-started})
        if best is None:
            raise ValueError(f'No sufficient stage certificate for axis {axis}')
        choices.append(best);certificates.append(winner);options.append(variants)
    for _ in range(2):
        for axis in reversed(error_order):
            rest,later=arb(0),arb(1)
            for i in error_order:
                q=choices[i]
                local=arb(0) if i==axis else axis_error(q['order'],q['cells'],q['ellipse'],
                                                      q['phase_derivative_bound'],q['hybrid_halfwidth'])
                rest=_lebesgue(q['order'])*rest+local
            for i in error_order[error_order.index(axis)+1:]:
                later*=_lebesgue(choices[i]['order'])
            allowance=(arb(tolerance)-rest)/later*arb('.95')
            if not allowance>0:
                continue
            target=float(np.nextafter(float(allowance.lower()),0.))
            if target<=0:
                continue
            for certificate in options[axis]:
                candidate=choose_axis(certificate,certificate['phase_derivative_bound'],target,
                                      max_order=choices[axis]['order'])
                if candidate['axis_nodes']<choices[axis]['axis_nodes']:
                    choices[axis],certificates[axis]=candidate,certificate
    report=grid_plan(choices,tolerance,permutation=error_order)
    report.update({'certificates':certificates,'trials_including_refusals':trials,
                   'time_range':list(time_range),'frequency_bounds':list(frequency_bounds),
                   'permutation':list(error_order),'construction_seconds':perf_counter()-begin,
                   'stage_aware_child_coordinates':True,'full_objective_complete':False})
    return report
