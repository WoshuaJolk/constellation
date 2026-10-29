"""Distance-conditioned photometric priorities, never an acceptance/size claim.

Keep bands separate. Unknown band, distance or magnitude remains an unsized lane.
The proxy ignores phase; diameters additionally assume V-equivalent photometry.
"""
import math


def brightness_priority(magnitude, band, heliocentric_au, observer_distance_au):
    values=(magnitude,heliocentric_au,observer_distance_au)
    if not isinstance(band,str) or not band.strip() or any(x is None or not math.isfinite(x) for x in values) or heliocentric_au<=0 or observer_distance_au<=0:
        return dict(lane='unsized',band=band,absolute_magnitude_proxy=None,
                    diameter_km_at_albedo_004_to_025=None)
    h=float(magnitude-5*math.log10(heliocentric_au*observer_distance_au))
    diameters=None
    if band=='V':
        diameters=[1329/math.sqrt(p)*10**(-h/5) for p in (.25,.04)]
    return dict(lane='photometry:'+band,band=band,absolute_magnitude_proxy=h,
                diameter_km_at_albedo_004_to_025=diameters,
                assumptions='Trial distances; zero phase correction. Rank within band only. V diameters assume albedo; no detection, orbit or size confirmation.')
