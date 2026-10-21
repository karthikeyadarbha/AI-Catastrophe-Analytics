"""Madhava-Newton series as rationalised in Ganita-Yuktibhasa (Ch. VI-VII) and quoted in Tantrasangraha.
All functions work in Indian units: R = trijya (radius, in arc-minutes, default 3437.74677...)."""
import math
R_EXACT = 10800.0/(2*math.pi)

def madhava_sine(s, R=R_EXACT, terms=12):
    """Rsine from arc s (same unit as R): s - s^3/(3!R^2) + s^5/(5!R^4) - ...  (GYB Ch.VII; TS 2.10b context)."""
    t, tot = s, s
    for k in range(1, terms):
        t *= -(s*s)/((2*k)*(2*k+1)*R*R); tot += t
    return tot

def madhava_cosine(s, R=R_EXACT, terms=12):
    """Rcosine: R - s^2/(2!R) + s^4/(4!R^3) - ..."""
    t, tot = R, R
    for k in range(1, terms):
        t *= -(s*s)/((2*k-1)*(2*k)*R*R); tot += t
    return tot

def madhava_arctan(jya_, kotijya_, R=R_EXACT, terms=12):
    """Arc from Rsine/Rcosine (GYB VI.6 chapikarana; TS II.206-209 as quoted):
    arc = R*[x - x^3/3 + x^5/5 ...], x = jya/kotijya, valid for jya<=kotijya (else use complement).
    Plain series; see madhava_arctan_corrected for the GYB VI.8 end correction."""
    x = jya_/kotijya_
    if abs(x) > 1: raise ValueError("use the complementary arc (koti-capa) when jya > kotijya")
    tot = 0.0; p = x
    for k in range(terms):
        tot += ((-1)**k)*p/(2*k+1); p *= x*x
    return R*tot

def madhava_arctan_corrected(x, terms=10):
    """Series with the end-correction term of GYB VI.8 (Madhava): for 2m+1 = last odd number,
    arc/R = x - x^3/3 + ... + (-1)^m x^(2m+1) * [ (n/2)/(n^2+1) ] with n = 2m+2 replacing 1/(2m+1)."""
    tot = 0.0
    for k in range(terms-1):
        tot += ((-1)**k)*(x**(2*k+1))/(2*k+1)
    m = terms-1
    n = 2*m+2
    tot += ((-1)**m)*(x**(2*m+1))*(n/2)/(n*n+1)
    return tot
