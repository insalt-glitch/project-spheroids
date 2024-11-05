#ifndef EVENTS_H
#define EVENTS_H

#include <array>
#include <bit>
#include <limits>
#include <cassert>
#define _USE_MATH_DEFINES
#include <math.h>
#include <gsl/gsl_errno.h>

#include "types.h"
#include "linalg.h"

struct SystemConstants;

struct IntegrateEvent {
    virtual int event(const f64 t,
        const double *const state,
        const double *const derivative,
        const void* const args) = 0;
    virtual ~IntegrateEvent() {};
};

struct NoneEvent : public IntegrateEvent {
    int event(
        const f64 t,
        const double *const state,
        const double *const derivative,
        const void* const args
    ) final {
        (void)(t);
        (void)(state);
        (void)(derivative);
        (void)(args);
        return GSL_SUCCESS;
    }
};

struct FixedPointEvent : public IntegrateEvent {
    size_t idx = 0;
    std::array<f64, 100> omega_buf;

    FixedPointEvent() {
        for (f64& x : omega_buf)
            x = 1;
    }

    int event(
        const f64 t,
        const double *const state,
        const double *const derivative,
        const void* const args
    ) final {
        (void)(t);
        (void)(derivative);
        (void)(args);
        omega_buf[idx] = sqrt(
            state[9] * state[9] +
            state[10] * state[10] +
            state[11] * state[11]);
        idx = (idx + 1) % omega_buf.size();
        double max = 0;
        for (f64 x : omega_buf)
            if (x > max)
                max = x;
        if (max < 1e-5)
            return GSL_EBADFUNC;
        return GSL_SUCCESS;
    }
};

struct OscillationEvent : public IntegrateEvent {
    size_t omega_idx = 0, theta_idx = 0;
    std::array<f64, 100> theta_buf;
    u16 omega_mask = 0xFFFF;

    f64 t_last;
    std::array<f64, 12> y_last;

    OscillationEvent() {
        for (f64& x : theta_buf)
            x = 100;
    }

    int event(
        const f64 t,
        const f64 *const state,
        const f64 *const derivative,
        const void* const args
    ) final {
        (void)(t);
        (void)(derivative);
        (void)(args);
        if (std::popcount(omega_mask) == 16 && state[10] < 0) {
            // Save last event time and state
            t_last = t;
            memcpy(y_last.data(), state, sizeof(f64) * 12);

            theta_idx = (theta_idx + 1) % theta_buf.size();
            theta_buf[theta_idx] = acos(state[8]);
            f64 max = 0;
            f64 min = std::numeric_limits<f64>::infinity();
            for (f64 x : theta_buf) {
                if (x > max)
                    max = x;
                if (x < min)
                    min = x;
            }
            assert(max == max && min == min);
            if (max - min < 1e-6)
                return GSL_EBADFUNC;

        }
        omega_idx = (omega_idx + 1) % 16;
        if (state[10] < 0)
            omega_mask &= ~(1 << omega_idx);
        else
            omega_mask |= 1 << omega_idx;
        return GSL_SUCCESS;
    }
};

enum class EventType {
    NONE,
    FIXED_POINT,
    OSCILLATION
};

IntegrateEvent* selectEventType(EventType type) {
    switch (type) {
        case EventType::NONE:
            return new NoneEvent();
        case EventType::FIXED_POINT:
            return new FixedPointEvent();
        case EventType::OSCILLATION:
            return new OscillationEvent();
        default:
            fprintf(stderr, "unknown event type");
            throw;
    }
}

#endif