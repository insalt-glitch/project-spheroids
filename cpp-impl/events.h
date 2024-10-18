#ifndef EVENTS_H
#define EVENTS_H

#include <array>
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
    std::array<f64, 100> buf;

    FixedPointEvent() {
        for (f64& x : buf)
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
        buf[idx] = sqrt(
            state[9] * state[9] +
            state[10] * state[10] +
            state[11] * state[11]);
        idx = (idx + 1) % buf.size();
        double max = 0;
        for (f64 x : buf)
            if (x > max)
                max = x;
        if (max < 1e-5)
            return GSL_EBADFUNC;
        return GSL_SUCCESS;
    }
};

enum class EventType {
    NONE,
    FIXED_POINT
};

IntegrateEvent* selectEventType(EventType type) {
    switch (type) {
        case EventType::NONE:
            return new NoneEvent();
        case EventType::FIXED_POINT:
            return new FixedPointEvent();
        default:
            fprintf(stderr, "unknown event type");
            throw;
    }
}

#endif