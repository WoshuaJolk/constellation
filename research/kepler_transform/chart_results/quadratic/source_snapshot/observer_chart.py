"""Observer-relative orbital coordinates; no conic-class restriction.

q = (inverse axial distance, fractional axial speed, north angular rate,
     east tangent coordinate, north tangent coordinate, east angular rate).

With retarded_epoch=True the position/velocity are specified at the emission
epoch seen by the reference observer at reception time zero. Angles then
describe that observed direction; rates still parameterize the emitted state.
The forward model still solves light time at every exposure. This is related to
attributable/inverse-distance orbit coordinates, not a novelty claim.
"""

import numpy as np

from .geometry_certificate import array_digest


class ObserverChart:
    def __init__(self, observer, velocity, normal, east, north, *, retarded_epoch=False):
        if not isinstance(retarded_epoch, bool):
            raise ValueError("retarded_epoch must be boolean")
        self.retarded_epoch = retarded_epoch
        self.observer, self.velocity = np.asarray(observer, float), np.asarray(velocity, float)
        self.axes = np.asarray([normal, east, north], float)
        if (self.observer.shape != (3,) or self.velocity.shape != (3,)
                or self.axes.shape != (3, 3)
                or not all(np.all(np.isfinite(x)) for x in [self.observer, self.velocity, self.axes])
                or not np.allclose(self.axes @ self.axes.T, np.eye(3), atol=2e-14, rtol=0)):
            raise ValueError("Finite observer state and orthonormal chart axes required")
        self.digest = array_digest(self.observer, self.velocity, self.axes, [float(retarded_epoch)])

    def states(self, parameters):
        q = np.asarray(parameters, float)
        if q.ndim != 2 or q.shape[1] != 6 or not np.all(np.isfinite(q)) or np.any(q[:, 0] <= 0):
            raise ValueError("Finite (N,6) chart parameters with positive inverse distance required")
        rho, eta, wy, x, y, wx = q.T
        normal, east, north = self.axes
        direction = normal+x[:, None]*east+y[:, None]*north
        position = self.observer+direction/rho[:, None]
        velocity = self.velocity+(eta[:, None]*direction+wx[:, None]*east+wy[:, None]*north)/rho[:, None]
        states = np.column_stack((position, velocity))
        if self.retarded_epoch:
            from magictools.orbits import C_AU_DAY
            from .universal import propagate
            delay = np.linalg.norm(direction, axis=1)/rho/C_AU_DAY
            states = propagate(states, delay[:, None])[:, 0]
        return states

    def parameters(self, states):
        states = np.asarray(states, float)
        if states.ndim != 2 or states.shape[1] != 6 or not np.all(np.isfinite(states)):
            raise ValueError("Finite (N,6) states required")
        if self.retarded_epoch:
            from .universal import apparent
            states = apparent(states, np.array([0.]), self.observer[None])[0][:, 0]
        r = (states[:, :3]-self.observer) @ self.axes.T
        v = (states[:, 3:]-self.velocity) @ self.axes.T
        if np.any(r[:, 0] <= 0):
            raise ValueError("States must be in front of the chart")
        rho = 1/r[:, 0]
        x, y, eta = r[:, 1]*rho, r[:, 2]*rho, v[:, 0]*rho
        return np.column_stack((rho, eta, v[:, 2]*rho-eta*y, x, y, v[:, 1]*rho-eta*x))

    def scaled_initial(self, parameters):
        """Generic scalar/Arb/Jet algebra for u=rho*r, w=rho*v.

        u'' = -mu*rho^3*u/|u|^3. Avoiding division by a wide distance
        interval is essential to keeping the variational certificate useful.
        """
        rho, eta, wy, x, y, wx = parameters
        normal, east, north = self.axes
        direction = [float(n)+x*float(e)+y*float(v) for n, e, v in zip(normal, east, north)]
        position = [rho*float(o)+d for o, d in zip(self.observer, direction)]
        velocity = [rho*float(o)+eta*d+wx*float(e)+wy*float(n)
                    for o, d, e, n in zip(self.velocity, direction, east, north)]
        return position+velocity

    def to_dict(self):
        return {"observer": self.observer.tolist(), "velocity": self.velocity.tolist(),
                "normal": self.axes[0].tolist(), "east": self.axes[1].tolist(),
                "north": self.axes[2].tolist(), "retarded_epoch": self.retarded_epoch}

    def path_offset(self, parameters):
        """Scaled reference light path, using generic scalar/Arb/Jet algebra."""
        x, y = parameters[3:5]
        normal, east, north = self.axes
        # Include the exact stored axes; do not assume exact orthonormality
        # from their floating-point validation.
        direction = [float(n)+x*float(e)+y*float(v) for n, e, v in zip(normal, east, north)]
        square = sum(d*d for d in direction)
        return square.sqrt() if hasattr(square, "sqrt") else np.sqrt(square)
