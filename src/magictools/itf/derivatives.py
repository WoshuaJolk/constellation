"""Analytic two-body/light-time Jacobians, shared by residual and optimizer calls.

Differentiate the universal Kepler equation implicitly. No finite-difference
propagation bank is needed for the six Cartesian initial-state coordinates.
"""
import math
import numpy as np
from .kernels import ARCSEC, C, MU, njit, stumpff


@njit(cache=True)
def universal_scalars(state, dt):
    """Allocation-free universal propagation; scalar tuple also supplies Jacobians."""
    rx, ry, rz, vx, vy, vz = state
    radius = math.sqrt(rx*rx+ry*ry+rz*rz)
    invalid = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    if radius < 1e-5 or not math.isfinite(radius):
        return invalid
    root_mu = math.sqrt(MU)
    rv = (rx*vx+ry*vy+rz*vz)/root_mu
    alpha = 2/radius-(vx*vx+vy*vy+vz*vz)/MU
    edge = root_mu*dt/radius
    for _ in range(80):
        c, ss = stumpff(alpha*edge*edge)
        value = rv*edge*edge*c+(1-alpha*radius)*edge**3*ss+radius*edge-root_mu*dt
        if not math.isfinite(value):
            return invalid
        if value*dt >= 0:
            break
        edge *= 2
    else:
        return invalid
    lo, hi = min(0., edge), max(0., edge)
    chi = edge
    for _ in range(100):
        z = alpha*chi*chi
        c, ss = stumpff(z)
        value = rv*chi*chi*c+(1-alpha*radius)*chi**3*ss+radius*chi-root_mu*dt
        radial = rv*chi*(1-z*ss)+(1-alpha*radius)*chi*chi*c+radius
        if not math.isfinite(radial) or radial <= 0:
            return invalid
        step = value/radial
        if abs(step) < 3e-14*max(1., abs(chi)):
            break
        if value < 0:
            lo = chi
        else:
            hi = chi
        proposed = chi-step
        chi = proposed if lo < proposed < hi else .5*(lo+hi)
    else:
        return invalid
    f, g = 1-chi*chi*c/radius, dt-chi**3*ss/root_mu
    x,y,zp = f*rx+g*vx, f*ry+g*vy, f*rz+g*vz
    rn = math.sqrt(x*x+y*y+zp*zp)
    if rn < 1e-5:
        return invalid
    fd = root_mu/(rn*radius)*(alpha*chi**3*ss-chi)
    gd = 1-chi*chi*c/rn
    return x,y,zp,fd*rx+gd*vx,fd*ry+gd*vy,fd*rz+gd*vz,chi,c,ss,radius,rv,alpha,radial,g


@njit(cache=True)
def geometry_into(state, dt, jacobian):
    values = universal_scalars(state, dt)
    return geometry_from_values(state, values, jacobian)


@njit(cache=True)
def geometry_from_values(state, values, jacobian):
    """Differentiate an already propagated state without solving Kepler again."""
    x,y,zp,vx,vy,vz,chi,c,ss,radius,rv,alpha,radial,g = values
    if not math.isfinite(x):
        jacobian[:] = np.nan
        return values
    z = alpha*chi*chi
    root_mu = math.sqrt(MU)
    f = 1-chi*chi*c/radius
    if abs(z) < 1e-3:
        cp = -1/24+z/360-z*z/13440+z**3/907200-z**4/95800320
        sp = -1/120+z/2520-z*z/120960+z**3/9979200-z**4/1245404160
    else:
        cp, sp = (1-z*ss-2*c)/(2*z), (c-3*ss)/(2*z)
    for k in range(6):
        dr = state[k]/radius if k < 3 else 0.
        drv = (state[k+3] if k < 3 else state[k-3])/root_mu
        da = -2*dr/(radius*radius)-(2*state[k]/MU if k >= 3 else 0.)
        fixed = drv*chi*chi*c-(da*radius+alpha*dr)*chi**3*ss+dr*chi
        fixed += (rv*chi**4*cp+(1-alpha*radius)*chi**5*sp)*da
        dchi = -fixed/radial
        dz = chi*chi*da+2*alpha*chi*dchi
        df = -(2*chi*dchi*c+chi*chi*cp*dz)/radius+chi*chi*c*dr/radius**2
        dg = -(3*chi*chi*dchi*ss+chi**3*sp*dz)/root_mu
        for j in range(3):
            jacobian[j,k] = df*state[j]+dg*state[j+3]
            if j == k:
                jacobian[j,k] += f
            if j+3 == k:
                jacobian[j,k] += g
    return values


@njit(cache=True)
def state_geometry(state, dt):
    jacobian = np.empty((3,6))
    values = geometry_into(state, dt, jacobian)
    return np.array(values[:3]), np.array(values[3:6]), jacobian


@njit(cache=True, nogil=True)
def residual_jacobian(state, offsets, observers, u, east, north, sigma):
    """Analytic derivatives of retarded topocentric tangent-plane residuals."""
    residual = np.empty(2*len(offsets))
    jacobian = np.empty((2*len(offsets),6))
    j = np.empty((3,6))
    factor = ARCSEC/sigma
    for i in range(len(offsets)):
        emitted = offsets[i]
        for _ in range(20):
            values = universal_scalars(state, emitted)
            dx,dy,dz = values[0]-observers[i,0],values[1]-observers[i,1],values[2]-observers[i,2]
            distance = math.sqrt(dx*dx+dy*dy+dz*dz)
            if not math.isfinite(distance) or distance < 1e-8:
                return np.full(2*len(offsets),1e12), np.zeros((2*len(offsets),6))
            # Solve emitted + range(emitted)/c = reception with its exact
            # derivative. Fixed-point iteration can need more than twenty steps
            # on fast radial branches even when a retarded solution exists.
            slope = 1+(dx*values[3]+dy*values[4]+dz*values[5])/(C*distance)
            updated = (emitted-(emitted-offsets[i]+distance/C)/slope
                       if math.isfinite(slope) and slope > 1e-8 else offsets[i]-distance/C)
            if not math.isfinite(updated):
                return np.full(2*len(offsets),1e12), np.zeros((2*len(offsets),6))
            if abs(updated-emitted) < 1e-11:
                break
            emitted = updated
        else:
            return np.full(2*len(offsets),1e12), np.zeros((2*len(offsets),6))
        # The converged light-time loop already evaluated this exact epoch.
        values = geometry_from_values(state, values, j)
        dx,dy,dz = values[0]-observers[i,0],values[1]-observers[i,1],values[2]-observers[i,2]
        vx,vy,vz = values[3],values[4],values[5]
        distance = math.sqrt(dx*dx+dy*dy+dz*dz)
        denominator = C*distance+dx*vx+dy*vy+dz*vz
        if not math.isfinite(distance) or distance < 1e-8 or abs(denominator) < 1e-8:
            return np.full(2*len(offsets),1e12), np.zeros((2*len(offsets),6))
        for k in range(6):
            shift = -(dx*j[0,k]+dy*j[1,k]+dz*j[2,k])/denominator
            j[0,k] += vx*shift
            j[1,k] += vy*shift
            j[2,k] += vz*shift
        radial = dx*u[i,0]+dy*u[i,1]+dz*u[i,2]
        for coordinate in range(2):
            tangent = east if coordinate == 0 else north
            transverse = dx*tangent[i,0]+dy*tangent[i,1]+dz*tangent[i,2]
            row = 2*i+coordinate
            residual[row] = math.atan2(transverse,radial)*factor
            norm = radial*radial+transverse*transverse
            if norm < 1e-20:
                return np.full(2*len(offsets),1e12), np.zeros((2*len(offsets),6))
            for k in range(6):
                du = u[i,0]*j[0,k]+u[i,1]*j[1,k]+u[i,2]*j[2,k]
                de = tangent[i,0]*j[0,k]+tangent[i,1]*j[1,k]+tangent[i,2]*j[2,k]
                jacobian[row,k] = (radial*de-transverse*du)/norm*factor
    return residual,jacobian


@njit(cache=True, nogil=True)
def solve_lm(seed, offsets, observers, u, east, north, sigma, max_evaluations=120):
    """Scaled damped Gauss-Newton with analytic Jacobians and bounded evaluations.

    The returned success flag requires numerical convergence. Astrometric and
    physical acceptance remain independent checks in fit_chain.
    """
    state = seed.copy()
    residual, jacobian = residual_jacobian(state, offsets, observers, u, east, north, sigma)
    cost = np.dot(residual, residual)
    damping, increase = 1e-3, 2.
    success = False
    refresh = True
    scales = np.zeros(6)
    normal, gradient = np.zeros((6,6)), np.zeros(6)
    for evaluation in range(max_evaluations):
        if not math.isfinite(cost) or cost >= 1e23:
            break
        if refresh:
            scales[:] = 0.
            normal[:] = 0.
            gradient[:] = 0.
            for row in range(len(residual)):
                for k in range(6):
                    scales[k] += jacobian[row,k]*jacobian[row,k]
            for k in range(6):
                scales[k] = max(math.sqrt(scales[k]), 1e-12)
            for row in range(len(residual)):
                for k in range(6):
                    value = jacobian[row,k]/scales[k]
                    gradient[k] += value*residual[row]
                    for l in range(k+1):
                        normal[k,l] += value*jacobian[row,l]/scales[l]
            for k in range(6):
                for l in range(k):
                    normal[l,k] = normal[k,l]
            refresh = False
        if np.max(np.abs(gradient)) <= 1e-7*max(1., math.sqrt(cost)):
            success = True
            break
        system = normal.copy()
        for k in range(6):
            system[k, k] += damping
        # Six-variable Cholesky avoids dispatching many tiny BLAS/LAPACK jobs.
        lower = np.zeros((6,6))
        positive = True
        for k in range(6):
            for l in range(k+1):
                value = system[k,l]
                for m in range(l):
                    value -= lower[k,m]*lower[l,m]
                if k == l:
                    if value <= 0:
                        positive = False
                        break
                    lower[k,l] = math.sqrt(value)
                else:
                    lower[k,l] = value/lower[l,l]
            if not positive:
                break
        if not positive:
            damping = max(damping*10, 1e-12)
            continue
        temporary, step_scaled = np.zeros(6), np.zeros(6)
        for k in range(6):
            value = -gradient[k]
            for l in range(k):
                value -= lower[k,l]*temporary[l]
            temporary[k] = value/lower[k,k]
        for k in range(5,-1,-1):
            value = temporary[k]
            for l in range(k+1,6):
                value -= lower[l,k]*step_scaled[l]
            step_scaled[k] = value/lower[k,k]
        step = step_scaled/scales
        candidate = state+step
        new_residual, new_jacobian = residual_jacobian(candidate, offsets, observers, u, east, north, sigma)
        new_cost = np.dot(new_residual, new_residual)
        predicted = -2*np.dot(gradient, step_scaled)-np.dot(step_scaled, normal@step_scaled)
        if new_cost < cost and predicted > 0 and math.isfinite(new_cost):
            ratio = (cost-new_cost)/predicted
            relative = (cost-new_cost)/max(cost, 1e-30)
            state, residual, jacobian = candidate, new_residual, new_jacobian
            refresh = True
            small = np.linalg.norm(step) <= 1e-9*(np.linalg.norm(state)+1e-9)
            cost = new_cost
            damping *= max(1/3, 1-(2*ratio-1)**3)
            damping = max(damping, 1e-15)
            increase = 2.
            if (relative < 1e-7 and predicted/max(cost, 1e-30) < 1e-7) or small:
                success = True
                break
        else:
            damping = min(damping*increase, 1e30)
            increase = min(2*increase, 1e12)
            if damping >= 1e25:
                # At floating-point resolution, a tiny gradient/step can no longer
                # lower cost. Otherwise this is a failed optimization.
                success = np.max(np.abs(gradient)) < 1e-5
                break
    return state, residual, jacobian, success
