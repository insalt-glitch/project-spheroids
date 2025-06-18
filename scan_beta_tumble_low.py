import numpy as np
import math
import dynamics
import c_dynamics
from pathlib import Path
from tqdm import tqdm
from functools import partial
import multiprocessing
from typing import Callable
from dynamics import SystemConstants as Constants
import h5py
from numpy.typing import NDArray
from enum import Enum
from pathlib import Path

class SearchGoal(Enum):
    BIFURCATION = "bifurcation"
    TUMBLING_TRANSITION = "tumbling_transition"

SEED = 0
NUM_BETAS = 10
SEARCH_DEPTH = 12
SAVE_TIME = (100, 105)
NUM_SAVED_STEPS = 50_000
MAX_INTEGRATION_TIME = 105
TOLERANCE = 1e-9
SEARCH_GOAL = SearchGoal.TUMBLING_TRANSITION

assert SAVE_TIME[0] > 5.0

def singleRun(const: Constants, rng: np.random.Generator) -> tuple[NDArray,NDArray]:
    # choose random initial conditions
    x0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([const.v_g, 0.0, 0.0])
    theta = np.pi / 2
    n0 = np.array([np.sin(theta), 0.0, np.cos(theta)])
    match SEARCH_GOAL:
        case SearchGoal.BIFURCATION:
            omega0 = rng.normal(loc=0.0, scale=1e-6, size=(3,)) + 1e-5 * np.sign(rng.normal(size=(3,)))
        case SearchGoal.TUMBLING_TRANSITION:
            omega0 = np.array([0, 0, 1e-5])
        case _:
            assert False, "Unknown search goal"
    y0 = np.concat([x0, v0, n0, omega0])
    # integrate dynamics
    return c_dynamics.solveDynamics(
        y0=y0,
        const=const,
        t_span=(0.0, MAX_INTEGRATION_TIME),
        t_eval=np.concat([
            np.linspace(0.0, 5.0, num=NUM_SAVED_STEPS),
            np.linspace(*SAVE_TIME, num=NUM_SAVED_STEPS),
        ]),
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    )

def bisectionSearch(
    low: float,
    high: float,
    setParameter: Callable[[float], Constants],
    search_depth: int,
    rng: np.random.Generator
):
    ts = np.empty(shape=(search_depth, NUM_SAVED_STEPS), dtype=np.float64)
    ys = np.empty(shape=(search_depth, NUM_SAVED_STEPS, 9), dtype=np.float64)
    volumes = np.empty(shape=(search_depth), dtype=np.float64)

    for i in range(search_depth):
        mid = (high + low) / 2
        volumes[i] = mid
        print(f"    iter {i+1:02} :: ({low:.2e}, {high:.2e}) --> {mid:.4e}", flush=True)
        t, y = singleRun(const=setParameter(volume=mid), rng=rng)
        match SEARCH_GOAL:
            case SearchGoal.BIFURCATION:
                ts[i] = t[NUM_SAVED_STEPS:]
                ys[i] = y[NUM_SAVED_STEPS:,3:]
                init_max_omega = np.max(np.abs(y[:NUM_SAVED_STEPS, 9:12]))
                final_max_omega = np.max(np.abs(y[NUM_SAVED_STEPS:, 9:12]))
                if init_max_omega > final_max_omega:
                    low = mid
                else:
                    high = mid
            case SearchGoal.TUMBLING_TRANSITION:
                ts[i] = t[NUM_SAVED_STEPS:]
                ys[i] = y[NUM_SAVED_STEPS:,3:]
                max_omega = np.max(y[NUM_SAVED_STEPS:, 9:12], axis=0)
                min_omega = np.min(y[NUM_SAVED_STEPS:, 9:12], axis=0)
                rel_offset = np.max(np.abs(max_omega + min_omega)) / np.max(max_omega - min_omega)
                if rel_offset < 1e-2:
                    low = mid
                else:
                    high = mid
            case _:
                assert False, "Unknown goal"

    return ts, ys, volumes

def setSimParameters(beta: float, volume: float):
    a_perp, a_para = dynamics.spheriodDimensionsFromBeta(beta, volume)
    parameters = Constants(
        a_para=a_para,
        a_perp=a_perp,
    )
    return parameters

if __name__ == "__main__":
    RNG = np.random.default_rng(SEED)

    bif_betas, bif_volumes = np.loadtxt("bifurcation_high.txt", delimiter=";")[:,::-1]
    bif_idx = np.linspace(0, bif_betas.size, num=20, dtype=int, endpoint=False)
    bif_betas = bif_betas[bif_idx]
    bif_volumes = bif_volumes[bif_idx]
    match SEARCH_GOAL:
        case SearchGoal.BIFURCATION:
            lower_bounds = bif_volumes / 2
            upper_bounds = bif_volumes * 2
        case SearchGoal.TUMBLING_TRANSITION:
            lower_bounds = bif_volumes
            upper_bounds = bif_volumes * 5
        case _:
            assert False, "Unknown goal"
    output_file = Path(f"data/{SEARCH_GOAL.value}_low_search_001.h5")
    assert not output_file.exists(), "Cannot overwrite output-file"

    with h5py.File(output_file, mode="x") as file:
        file.attrs["seed"] = SEED
        file.attrs["tolerance"] = TOLERANCE
        file.attrs["goal"] = SEARCH_GOAL.value
        file.create_dataset("betas", data=bif_betas)
        file.create_dataset("analytic_volumes", data=bif_volumes)
        file.create_dataset("search_bounds", data= np.concat([lower_bounds, upper_bounds]))

    for i, (beta, lower, upper) in enumerate(zip(
        bif_betas, lower_bounds, upper_bounds
    )):
        print(f"{i+1}/{bif_betas.size} :: beta={beta:.2f}", flush=True)
        time, state, volume = bisectionSearch(
            low=lower,
            high=upper,
            setParameter=partial(setSimParameters, beta=beta),
            search_depth=SEARCH_DEPTH,
            rng=RNG,
        )
        with h5py.File(output_file, mode="r+") as file:
            group = file.create_group(f"{i}")
            group.attrs["beta"] = beta
            group.attrs["bounds"] = (lower, upper)
            group.attrs["description"] = "Time and state values for each search point. Last search point is the final result."
            group.create_dataset("time", data=time)
            group.create_dataset("state", data=state).attrs["state_variables"] = "v_x v_y v_z n_x n_y n_z omega_x omega_y omega_z"
            group.create_dataset("volume", data=volume)
