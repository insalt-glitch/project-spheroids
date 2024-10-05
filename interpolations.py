import numpy as np
from scipy import optimize
import dynamics

def selfConsistencyEqProlateC_F(C_F, A_perp, Re_p0, beta):
    c_d = [-0.007, 1.0, 1.17, -0.07, 0.047, 1.14, 0.7, -0.008]
    return + 1 - np.sqrt(1 + 3 * A_perp * C_F * Re_p0 / (2 * beta)) \
        + 2 * (+ 0.18277810531719724 * beta ** 0.229 * ((- 2 + np.sqrt(4 + 6 * A_perp \
            * C_F * Re_p0 / beta)) / (A_perp * C_F)) ** 0.687 + 0.75 ** (-c_d[6] - c_d[7] \
            * np.log(beta)) * c_d[4] * (beta ** (1 / 3) * (- 2 + np.sqrt(4 + 6 * A_perp \
            * C_F * Re_p0 / beta)) / (A_perp * C_F)) ** (c_d[6] + c_d[7] * np.log(beta)) \
            * np.log(beta) ** c_d[5] \
        )

def selfConsistencyEqOblateC_F(C_F, A_para, Re_p0, beta):
    return (
        A_para * (2 - np.sqrt(2) * np.sqrt(2 + 3 * A_para * C_F * Re_p0)) \
        + 0.7311124212687891 * beta ** 96.47233333333332  * ((-2 + np.sqrt(4 + 6 * A_para \
        * C_F * Re_p0)) / (A_para * C_F)) ** 0.687 + 1.453237823359436 \
        * (1 - beta) ** 0.4374 * beta ** 0.5837333333333332 * ((-2 + np.sqrt(4 + 6 * A_para \
        * C_F * Re_p0)) / (A_para * C_F)) ** 0.7512
    )

if __name__ == '__main__':
    num_coeffs = 20
    aspect_ratio_interval = [0.1, 0.8, 1.2, ]
    const = dynamics.SystemConstants()
    for ratios in [
        np.linspace(0.1, 0.8, num=num_coeffs),
        np.linspace(1.2, 6, num=num_coeffs),
    ]:
        C_F_arr = np.empty_like(ratios)
        for i, beta in enumerate(ratios):
            a_perp, a_para = dynamics.spheriodDimensionsFromBeta(
                beta, const.particle_volume
            )
            const = dynamics.SystemConstants(a_perp=a_perp, a_para=a_para)
            Re_p0 = dynamics.particleReynoldsNumber(
                const.a_perp, const.a_para, const.W_approx, const.nu
            )
            func = selfConsistencyEqProlateC_F if beta >= 1 else selfConsistencyEqOblateC_F
            A_coeff = const.A_perp if beta >= 1 else const.A_para
            C_F_arr[i] = optimize.least_squares(
                func, x0=1.0, args=(A_coeff, Re_p0, beta),
                bounds=(0.0, np.inf)).x[0]
        print(C_F_arr)
