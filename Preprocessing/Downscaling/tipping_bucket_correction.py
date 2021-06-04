from math import e

def pce_correct(U, t2m, tp, measurement_h=2):
    """Transfer function to correct tipping bucket data for solid precipitation undercatch.
    Divides passed precipitation data by a wind dependent catch efficiency.
    Refers to EQ2 & Table 3 from Kochendorfer et.al. 2020. """

    a_gh_mix = 0.726    # gh = at gauging bucket height
    b_gh_mix = 0.0495   # mix = mixed precip (2° ≥ Tair ≥ −2°C)
    a_gh_solid = 0.701  # solid = solid precip (Tair < −2°C)
    b_gh_solid = 0.227
    U_thresh_gh = 6.1   # maximum wind speed

    a_10m_mix = 0.722   # 10m = at 10m height
    b_10m_mix = 0.0354
    a_10m_solid = 0.7116
    b_10m_solid = 0.1925
    U_thresh_10m = 8

    def cond_solid(U_thresh):
        return (U <= U_thresh) & (t2m <= 271.15)

    def cond_mix(U_thresh):
        return (U <= U_thresh) & (275.15 >= t2m) & (t2m >= 271.15)

    if measurement_h < 7:
        cond_solid = cond_solid(U_thresh_gh)
        cond_mix = cond_mix(U_thresh_gh)

        tp[cond_solid] = tp[cond_solid] / ((a_gh_solid) * e ** (-b_gh_solid * U[cond_solid]))
        tp[cond_mix] = tp[cond_mix] / ((a_gh_mix) * e ** (-b_gh_mix * U[cond_mix]))

    else:
        cond_solid = cond_solid(U_thresh_10m)
        cond_mix = cond_mix(U_thresh_10m)

        tp[cond_solid] = tp[cond_solid] / ((a_10m_solid) * e ** (-b_10m_solid * U[cond_solid]))
        tp[cond_mix] = tp[cond_mix] / ((a_10m_mix) * e ** (-b_10m_mix * U[cond_mix]))

    return tp

