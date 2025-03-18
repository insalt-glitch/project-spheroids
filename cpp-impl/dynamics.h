#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "types.h"
#include "linalg.h"

extern "C" struct SystemConstants{
    f64 beta, curly_A_F, curly_A_T;
    f64 A_para, A_perp, C_para, C_perp, J_para, J_perp, Re_p0;
    f64 _fac_Re_p, _fac1_v_g_star, _fac2_v_g_star;
    f64 _C_T_prolate_c0, _C_T_prolate_c1, _C_T_prolate_c2;
    f64 _C_T_oblate_c0;
};

constexpr f64 cd[] = {-0.007, 1.0, 1.17, -0.07, 0.047, 1.14, 0.7, -0.008};

static f64 selfConsistencyEqProlateC_F(
    const f64 C_F, const f64 A_perp, const f64 Re_p0,
    const f64 beta
) {
    return 1.0 - sqrt(1.0 + (3.0 * A_perp*C_F*Re_p0)/(2.0*beta)) + 2.0*(0.18277810531719724*pow(beta,0.229)*pow((-2 + sqrt(4 + (6*A_perp*C_F*Re_p0)/beta))/(A_perp*C_F),0.687) + pow(0.75, -cd[6] - cd[7]*log(beta))*cd[4]*pow((pow(beta, 0.3333333333333333)*(-2.0 + sqrt(4.0 + (6.0*A_perp*C_F*Re_p0)/beta)))/(A_perp*C_F), cd[6] + cd[7]*log(beta)) * pow(log(beta), cd[5]));
}

static f64 selfConsistencyEqOblateC_F(
    const f64 C_F, const f64 A_para, const f64 Re_p0,
    const f64 beta
) {
    return A_para*(2.0 - 1.4142135623730951*sqrt(2.0 + 3.0*A_para*C_F*Re_p0)) + 0.7311124212687891*pow(beta,96.47233333333332)*pow((-2.0 + sqrt(4.0 + 6.0*A_para*C_F*Re_p0))/(A_para*C_F), 0.687) + 1.453237823359436*pow(1.0 - beta,0.4374)*pow(beta,0.5837333333333332)*pow((-2.0 + sqrt(4.0 + 6.0*A_para*C_F*Re_p0))/(A_para*C_F), 0.7512);
}

extern "C" f64 correctionFactorStokesForce(const f64 Re_p0, const SystemConstants *const sc) {
    // Re_0 upper bound of the actual reynolds number
    // The correction factor C_F for the stokes force for Reynolds numbers > 1
    // full_solve: Whether to solve the self-consistency equation for C_F otherwise an
    //   interpolation formula is used.
    if (Re_p0 <= 1.0) {
        return 1.0;
    }
    f64 C_F;

    if (sc->beta >= 1.0) {  // beta > 1 -> prolate spheroid
        // The following formulas are discussed in Fröhlich JFM 901 (2020):
        //   https://doi.org/10.1017/jfm.2020.482
        // Consider rod-like particles aligned with the steady state direction
        // (phi=pi/2), i.e. consider the drag coefficient (Page 19 - Section 3.3.1 - Eq. 3.4b)
        //   C_{D,90}(Re,beta)=C_{D,Stokes,90}(Re,beta)*f_{d,90}(Re,beta)
        //   C_{D,Stokes,90}(Re,beta) is the analytical drag coefficient (Page 30 - Eq. B5)
        // Take the correction function (Page 30 - Eq. B6b) f_{d,90}(Re,beta) then
        // in our model, we have vgDot=g-A_perp v_g/taup[1+C_F*3/8*ap/beta*A_perp*vg/nu]
        // => f_{d,90}(Re,beta)=1+0.15*Re_JFM^0.687+c_{d,5}*log(beta)^c_{d,6}*Re_JFM^(c_{d,7}+c_{d,8}*log(beta))
        //                     =1+C_F*3*A_perp*Rep/(8*beta)
        // => C_F=8*beta*(0.15*Re_JFM^0.687+c_{d,5}*log(beta)^c_{d,6}*Re_JFM^(c_{d,7}+c{d,8}*log(beta)))/(3*A_perp*Rep)
        C_F = bisectionMethod(selfConsistencyEqProlateC_F, sc->A_perp, Re_p0, sc->beta);
    } else {  // beta < 1 -> oblate spheroid
        // Use Ouchene (2020) for interpolated coefficient : https://doi.org/10.1063/5.0011618
        // Consider disk-like particles aligned with the steady state direction
        // (phi=0), i.e. consider the drag coefficient
        C_F = bisectionMethod(selfConsistencyEqOblateC_F, sc->A_para, Re_p0, sc->beta);
    }
    return C_F;
}

extern "C" f64 correctionFactorTorque(const f64 Re_p, const SystemConstants *const sc) {
    if (Re_p <= 1.0) {
        return 1.0;
    }
    f64 C_T;
    if (sc->beta > 1.0) {  // beta > 1 -> prolate spheroid
        C_T = sc->_C_T_prolate_c1 * pow(Re_p, -0.162) + sc->_C_T_prolate_c2 * pow(Re_p, -sc->_C_T_prolate_c0);
    } else {  // beta < 1 -> oblate spheroid
        C_T = sc->_C_T_oblate_c0 * pow(Re_p, -0.146);
    }
    return C_T;
}

extern "C" int spheriodDynamics(
    const f64 t,
    const f64 *const state,
    f64 *const derivative,
    void* args// void* args
) {
    static constexpr Vec3 g_hat = { .x = 1.0, .y = 0.0, .z = 0.0 };
    (void)(t);
    const SystemConstants *const sc = (SystemConstants*)args;

    // extract state variables
    const Vec3 v     = { .x = state[3], .y = state[4], .z =state[5] };
          Vec3 n     = { .x = state[6], .y = state[7], .z =state[8] };
    const Vec3 omega = { .x = state[9], .y = state[10], .z =state[11] };
    n = n / norm(n);

    const f64 v_mag = norm(v);
    const Vec3 v_hat = v / v_mag;
    // A = Translation resistance tensor
    // C = Rotation resistance tensor
    const Mat3 A = sc->A_perp * identity() + (sc->A_para - sc->A_perp) * outer(n, n);
    const Mat3 C = sc->C_perp * identity() + (sc->C_para - sc->C_perp) * outer(n, n);
    // Stokes force + correction for higher particle Reynolds numbers (Re_p ~ 1 - 30)
    const Vec3 F_h0 = (-1.0) * matmul(A, v);
    const Vec3 F_h1 = (-1.0) * sc->curly_A_F * v_mag *
        matmul((3.0 * A - dot(v_hat, matmul(A , v_hat)) * identity()), matmul(A, v));
    // Torque + correction for higher particle Reynolds numbers (Re_p ~ 1 - 30)
    const Vec3 T_h0 = (-1.0) * matmul(C, omega);
    const Vec3 T_h1 = sc->curly_A_T * dot(n, v) * cross(n, v);
    // Compute correction factors
    const f64 C_F = correctionFactorStokesForce(sc->Re_p0, sc);
    const f64 v_g_star = sc->_fac1_v_g_star * (sqrt(1 + sc->_fac2_v_g_star * C_F) - 1) / C_F;
    f64 Re_p = (C_F < 1e-5) ? (sc->Re_p0) : (sc->_fac_Re_p * v_g_star);
    const f64 C_T = correctionFactorTorque(Re_p, sc);
    // Full terms for torque and stokes force
    const Vec3 F_h = F_h0 + C_F * F_h1;
    const Vec3 T_h = T_h0 + C_T * T_h1;
    // Particle interia tensor
    const Mat3 J_pinverse = (1.0 / (sc->J_perp * sc->J_para)) * (
        (sc->J_perp - sc->J_para) * outer(n, n) + sc->J_para * identity()
    );
    const Mat3 dJ_pdt = (sc->J_para - sc->J_perp) * (
        outer(n, cross(omega, n)) + outer(cross(omega, n), n)
    );
    // Derivatives of the system variables
    const Vec3 dxdt = v;
    const Vec3 dvdt = F_h + g_hat;
    const Vec3 dndt = cross(omega, n);
    const Vec3 domegadt = matmul(J_pinverse, (T_h - matmul(dJ_pdt, omega)));
    // copy into result vector
    derivative[0+0] = dxdt.x;
    derivative[0+1] = dxdt.y;
    derivative[0+2] = dxdt.z;
    derivative[3+0] = dvdt.x;
    derivative[3+1] = dvdt.y;
    derivative[3+2] = dvdt.z;
    derivative[6+0] = dndt.x;
    derivative[6+1] = dndt.y;
    derivative[6+2] = dndt.z;
    derivative[9+0] = domegadt.x;
    derivative[9+1] = domegadt.y;
    derivative[9+2] = domegadt.z;
    return GSL_SUCCESS;
}

#endif