"""Compiled parsing and all-conic two-body astrometry kernels."""
import math
import numpy as np
try:
    from numba import njit
except ImportError as exc:
    raise ImportError('solve_itf requires acceleration: pip install -e ".[itf]"') from exc

MU = 0.0002959122082855911
C = 173.144632674240
ARCSEC = 206264.80624709636


@njit(cache=True)
def number(row, lo, hi):
    value, fraction, scale, sign, digits = 0., False, 1., 1., 0
    for i in range(lo, hi):
        c = row[i]
        if c == 32:
            continue
        if c == 45 and digits == 0:
            sign = -1.
        elif c == 43 and digits == 0:
            pass
        elif c == 46 and not fraction:
            fraction = True
        elif 48 <= c <= 57:
            value = value * 10 + c - 48
            digits += 1
            if fraction:
                scale *= 10
        else:
            return np.nan
    return sign * value / scale if digits else np.nan


@njit(cache=True)
def parse_numeric(raw):
    values = np.zeros((len(raw), 3))
    status = np.zeros(len(raw), np.int8)
    for i in range(len(raw)):
        row = raw[i]
        # Satellite, roving and radar records need their separate observer model.
        if row[14] in (83, 115, 86, 118, 82, 114):
            status[i] = 2
            continue
        y, m, d = number(row, 15, 19), number(row, 20, 22), number(row, 23, 32)
        h, rm, rs = number(row, 32, 34), number(row, 35, 37), number(row, 38, 44)
        dd, dm, ds = number(row, 45, 47), number(row, 48, 50), number(row, 51, 56)
        if not (1600 <= y <= 2200 and y == int(y) and 1 <= m <= 12 and m == int(m)
                and 1 <= d < 32 and 0 <= h < 24 and 0 <= rm < 60 and 0 <= rs < 60
                and 0 <= dd <= 90 and 0 <= dm < 60 and 0 <= ds < 60
                and row[44] in (43, 45, 32)):
            status[i] = 1
            continue
        year, month = int(y), int(m)
        mdays = (31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if d >= mdays[month-1]+1:
            status[i] = 1
            continue
        a = (14-month)//12
        yy, mm = year+4800-a, month+12*a-3
        mjd = int(d)+(153*mm+2)//5+365*yy+yy//4-yy//100+yy//400-32045-2400001+(d-int(d))
        dec = dd+dm/60+ds/3600
        if dec > 90:
            status[i] = 1
            continue
        values[i, 0] = mjd
        values[i, 1] = (h+rm/60+rs/3600)*15
        values[i, 2] = -dec if row[44] == 45 else dec
    return values, status


@njit(cache=True)
def stumpff(z):
    if abs(z) < 1e-3:
        return (.5-z/24+z*z/720-z**3/40320+z**4/3628800,
                1/6-z/120+z*z/5040-z**3/362880+z**4/39916800)
    if z > 0:
        x = math.sqrt(z)
        return 2*math.sin(x/2)**2/z, (x-math.sin(x))/x**3
    x = math.sqrt(-z)
    if x > 500:
        return np.nan, np.nan
    return (math.cosh(x)-1)/(-z), (math.sinh(x)-x)/x**3


@njit(cache=True)
def position(state, dt):
    r = math.sqrt(np.dot(state[:3], state[:3]))
    if r < 1e-5:
        return np.full(3, np.nan)
    rv = np.dot(state[:3], state[3:])/math.sqrt(MU)
    alpha = 2/r-np.dot(state[3:], state[3:])/MU
    edge = math.sqrt(MU)*dt/r
    for _ in range(80):
        c, s = stumpff(alpha*edge*edge)
        f = rv*edge*edge*c+(1-alpha*r)*edge**3*s+r*edge-math.sqrt(MU)*dt
        if not math.isfinite(f):
            return np.full(3, np.nan)
        if f*dt >= 0:
            break
        edge *= 2
    else:
        return np.full(3, np.nan)
    lo, hi = min(0., edge), max(0., edge)
    chi = (lo+hi)/2
    for _ in range(100):
        z = alpha*chi*chi
        c, s = stumpff(z)
        f = rv*chi*chi*c+(1-alpha*r)*chi**3*s+r*chi-math.sqrt(MU)*dt
        df = rv*chi*(1-z*s)+(1-alpha*r)*chi*chi*c+r
        if not math.isfinite(df) or df <= 0:
            return np.full(3, np.nan)
        step = f/df
        if abs(step) < 3e-14*max(1., abs(chi)):
            break
        if f < 0:
            lo = chi
        else:
            hi = chi
        new = chi-step
        chi = new if lo < new < hi else (lo+hi)/2
    else:
        return np.full(3, np.nan)
    return (1-chi*chi*c/r)*state[:3]+(dt-chi**3*s/math.sqrt(MU))*state[3:]


@njit(cache=True)
def directions(state, offsets, observers):
    result = np.empty((len(offsets), 3))
    for i in range(len(offsets)):
        emission = offsets[i]
        delta = np.zeros(3)
        for _ in range(20):
            delta = position(state, emission)-observers[i]
            distance = math.sqrt(np.dot(delta, delta))
            updated = offsets[i]-distance/C
            if abs(updated-emission) < 1e-11:
                break
            emission = updated
        else:
            result[i] = np.nan
            continue
        result[i] = delta/distance
    return result


@njit(cache=True)
def summaries(t, u, starts, ends):
    out = np.zeros((len(starts), 9))
    for k in range(len(starts)):
        a, b = starts[k], ends[k]
        mean_t = np.mean(t[a:b])
        mean_u = np.zeros(3)
        for j in range(a, b):
            mean_u += u[j]
        mean_u /= b-a
        slope, denom = np.zeros(3), 0.
        for j in range(a, b):
            dt = t[j]-mean_t
            slope += dt*(u[j]-mean_u)
            denom += dt*dt
        if denom > 0:
            slope /= denom
        ss = 0.
        for j in range(a, b):
            residual = u[j]-mean_u-slope*(t[j]-mean_t)
            ss += np.dot(residual, residual)
        out[k, 0] = mean_t
        out[k, 1:4] = mean_u/np.linalg.norm(mean_u)
        out[k, 4:7] = slope
        out[k, 7] = math.sqrt(ss/(b-a))*ARCSEC
        out[k, 8] = t[b-1]-t[a]
    return out
