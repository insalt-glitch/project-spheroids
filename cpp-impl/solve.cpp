#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <cstring>
#include <vector>

#include "types.h"
#include "dynamics.cpp"

constexpr size_t DIMENSION = 12;

extern "C" void solveDynamics(
    f64 **const y_eval,
    size_t *num_eval,
    const f64 *const t_eval,
    const size_t len_t_eval,
    f64 *const y0,
    SystemConstants *const constants,
    const f64 rel_tol,
    const f64 abs_tol
) {
    assert(y_eval != NULL);
    assert(y_eval[0] != NULL);
    assert(num_eval != NULL);
    assert(len_t_eval > 0);
    assert(t_eval != NULL);
    assert(y0 != NULL);
    assert(constants != NULL);
    assert(rel_tol > 0);
    assert(abs_tol > 0);

    gsl_odeiv2_system sys = {
        .function = systemDynamics,
        .jacobian = NULL,
        .dimension = DIMENSION,
        .params = constants
    };
    gsl_odeiv2_driver *d =
        gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
                                      1e-6, rel_tol, abs_tol);
    assert(d != NULL);

    f64 t = t_eval[0];
    f64 y[DIMENSION];
    memcpy(y, y0, sizeof(f64) * DIMENSION);

    int status;
    size_t i = 0;
    for (; i < len_t_eval; ++i) {
        status = gsl_odeiv2_driver_apply(d, &t, t_eval[i], y);
        memcpy(y_eval[i], y, sizeof(f64) * DIMENSION);
        if (status != GSL_SUCCESS) break;
    }
    *num_eval = i;
    gsl_odeiv2_driver_free(d);
}
