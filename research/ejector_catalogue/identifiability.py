"""Construct an exact missing-RV ambiguity; this is established kinematics.

In an inertial, force-free frame with observer at zero, take observed positions
r_i=d_i u_i and proper-motion velocities v_perp_i=-P_i w, P_i=I-u_i u_i^T.
For ANY positive lookback T, set v_i(T)=r_i/T-w. Every star then satisfies
r_i-T*v_i(T)=T*w, while P_i*v_i(T)=-P_i*w is independent of T. Distinct ages,
origins and radial velocities therefore have identical perfect 5D astrometry.
This is a counterexample to unique recovery from 5D data, not a new theorem or
an argument against probabilistic inference with additional measurements.
"""
import numpy as np
from magictools.galactic import KMS_TO_PC_MYR


def ambiguity():
    positions = np.array([[1000.,200.,50.],[500.,1500.,-100.],[-400.,800.,300.]])
    direction = positions / np.linalg.norm(positions,axis=1)[:,None]
    w = np.array([120.,-80.,30.]) # pc/Myr
    histories=[]
    for age in [2.,5.]:
        velocity=positions/age-w
        radial=np.sum(velocity*direction,axis=1)
        transverse=velocity-radial[:,None]*direction
        histories.append(dict(lookback_myr=age,origin_pc=(age*w).tolist(),
            radial_velocity_kms=(radial/KMS_TO_PC_MYR).tolist(),
            transverse_velocity_kms=(transverse/KMS_TO_PC_MYR).tolist(),
            convergence_error_pc=float(np.linalg.norm(positions-age*velocity-age*w,axis=1).max())))
    return dict(histories=histories,max_transverse_velocity_difference_kms=float(np.max(np.abs(
        np.array(histories[0]['transverse_velocity_kms'])-histories[1]['transverse_velocity_kms']))),
        conclusion='Identical perfect 5D astrometry can admit distinct common ejection ages and sites; missing RV cannot be resolved by declaring a Gaussian prior to be a measurement.')


if __name__=='__main__':
    import json
    print(json.dumps(ambiguity(),indent=2))
