#ifndef SOLVE_H
#define SOLVE_H
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <cstring>
#include <vector>

#include "types.h"
#include "events.h"
#include "dynamics.h"

constexpr size_t DIMENSION = 12;
typedef int (*IntegrateFunction)(double t, const double *y, double *dydt, void *params);

struct ParameterPack {
    SystemConstants* constants;
    IntegrateEvent* event_ptr;
};

template<IntegrateFunction integrateFunc>
static int integrateWithEvent(double t, const double *state, double *derivative, void* arg) {
    ParameterPack* param_pack = (ParameterPack*) arg;
    int status = integrateFunc(t, state, derivative, param_pack->constants);
    if (status != GSL_SUCCESS)
        return status;
    return param_pack->event_ptr->event(t, state, derivative, param_pack->constants);
}

extern "C" void solveDynamics(
    f64 **const y_eval,
    size_t *num_eval,
    f64 *const t_span,
    const f64 *const t_eval,
    const size_t len_t_eval,
    f64 *const y0,
    SystemConstants *const constants,
    const f64 rel_tol,
    const f64 abs_tol,
    EventType event_type
) {
    assert(y_eval != nullptr);
    assert(y_eval[0] != nullptr);
    assert(num_eval != nullptr);
    assert(y0 != nullptr);
    assert(constants != nullptr);
    assert((t_eval != nullptr && len_t_eval >0) || t_span != nullptr);
    assert(rel_tol > 0);
    assert(abs_tol > 0);

    // TODO: This could also be a temporary solution, but we want to send the average (or the last) value of the oscillation event buffer back to python to save it.
    IntegrateEvent* event_ptr = selectEventType(event_type);
    ParameterPack params = {
        .constants = constants,
        .event_ptr = event_ptr
    };
    gsl_odeiv2_system sys = {
        .function = integrateWithEvent<&spheriodDynamics>,
        .jacobian = nullptr,
        .dimension = DIMENSION,
        .params = &params
    };
    gsl_odeiv2_driver *d =
        gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
                                      1e-6, rel_tol, abs_tol);
    assert(d != nullptr);

    f64 y[DIMENSION];
    memcpy(y, y0, sizeof(f64) * DIMENSION);

    int status;
    // t_span gets ignored if t_eval is set.
    if (t_eval != nullptr) {
        f64 t = t_eval[0];
        size_t i = 0;
        for (; i < len_t_eval; ++i) {
            status = gsl_odeiv2_driver_apply(d, &t, t_eval[i], y);
            memcpy(y_eval[i], y, sizeof(f64) * DIMENSION);
            if (status != GSL_SUCCESS) break;
        }
        *num_eval = i;
    } else {
        f64 t_start = t_span[0];
        f64 t_end = t_span[1];
        gsl_odeiv2_driver_apply(d, &t_start, t_end, y);
        // TODO: Temporary hack to return the state at the last event
        if (event_type == EventType::OSCILLATION) {
            OscillationEvent* osc_event = (OscillationEvent*)event_ptr;
            memcpy(y_eval[0], osc_event->y_last.data(), sizeof(f64) * DIMENSION);
            t_span[1] = osc_event->t_last;
        } else {
            memcpy(y_eval[0], y, sizeof(f64) * DIMENSION);
            t_span[1] = t_start;
        }
        *num_eval = 1;
    }

    gsl_odeiv2_driver_free(d);
    delete params.event_ptr;
}

#endif