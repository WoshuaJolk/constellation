"""Additional distant starts; ordinary all-conic starts and CPU checks remain."""
import time
from dataclasses import replace
import numpy as np
import gpu_stream_adapter as base
from gpu_stream_adapter_v2 import StreamGPUBatchV2
from gpu_batch_host import prepare_bank
from magictools.itf.distant import distant_seeds

DISTANCES=(30.,50.,80.,120.,200.,350.,600.,1000.,1600.)


def augment_jobs(geometry,jobs):
    out=[];count=0;unsupported=0
    for key,settings in jobs:
        t,u,observers,_=geometry.get(key)
        extra,receipt=distant_seeds(t,u,observers,distances=DISTANCES,radial_points=3)
        out.append((key,dict(settings,extra_seeds=tuple(settings.get('extra_seeds',()))+tuple(extra))))
        count+=len(extra);unsupported+=sum(r['starts']==0 for r in receipt['distances'])
    return out,dict(extra_distant_starts=count,unsupported_chord_distances=unsupported)


def prepare(jobs):
    started=time.perf_counter();augmented,counts=augment_jobs(base._GEOMETRY,jobs)
    bank,plans,groups,epochs,bounds,limit=prepare_bank(base._GEOMETRY,augmented,compact_geometry=True)
    plans=[replace(p,seeds=()) if p is not None else None for p in plans]
    receipt=base.store_prepared((bank,plans,groups,epochs,bounds,limit),base._SHARED);receipt['distant_counts']=counts
    return receipt,time.perf_counter()-started


class DistantGPUAdapter(StreamGPUBatchV2):
    prepare_task=staticmethod(prepare)
    def _finish(self,prepared,jobs,out,receipt,planning,solve_receipt):
        rows,record=super()._finish(prepared,jobs,out,receipt,planning,solve_receipt)
        record.update(receipt.get('distant_counts',{}));return rows,record
