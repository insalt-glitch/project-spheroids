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

SEED = 0
NUM_OMEGAS = 20
SEARCH_DEPTH = 12
SAVE_TIME = (100, 105)
NUM_SAVED_STEPS = 50_000
MAX_INTEGRATION_TIME = 105
TOLERANCE = 1e-9

assert SAVE_TIME[0] > 5.0

def singleRun(const: Constants, rng: np.random.Generator, omega0: NDArray) -> tuple[NDArray,NDArray]:
    # choose random initial conditions
    x0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([const.v_g, 0.0, 0.0])
    theta = np.pi / 2
    n0 = np.array([np.sin(theta), 0.0, np.cos(theta)])
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
    runFunc: Callable[[Constants, np.random.Generator], tuple[NDArray,NDArray]],
    search_depth: int,
    rng: np.random.Generator
):
    ts = np.empty(shape=(search_depth, NUM_SAVED_STEPS), dtype=np.float64)
    ys = np.empty(shape=(search_depth, NUM_SAVED_STEPS, 9), dtype=np.float64)
    volumes = np.empty(shape=(search_depth), dtype=np.float64)

    for i in range(search_depth):
        mid = (high + low) / 2
        volumes[i] = mid
        print(f"    iter {i+1:02} :: ({low:.4e}, {high:.4e}) --> {mid:.6e}")
        t, y = runFunc(const=setParameter(volume=mid), rng=rng)
        ts[i] = t[NUM_SAVED_STEPS:]
        ys[i] = y[NUM_SAVED_STEPS:,3:]
        max_omega = np.max(y[NUM_SAVED_STEPS:, 9:12], axis=0)
        min_omega = np.min(y[NUM_SAVED_STEPS:, 9:12], axis=0)
        rel_offset = np.max(np.abs(max_omega + min_omega)) / np.max(max_omega - min_omega)
        if rel_offset < 1e-2:
            low = mid
        else:
            high = mid

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

    # search at beta = 0.1 for now (because the oscillation frequency is low there)
    bif_beta, bif_volume = np.loadtxt("bifurcation_high.txt", delimiter=";")[:,-1]

    p = np.linspace(1e-4, 1-1e-4, NUM_OMEGAS)
    omega0s = np.vstack([
        np.zeros(shape=(NUM_OMEGAS)),
        np.log(p/(1-p)) / (0.01861181528410995) + 500,
        np.zeros(shape=(NUM_OMEGAS)),
    ]).T

    lower_bound = 3.8e-10
    upper_bound = 4.5e-10
    output_file = Path(f"data/omega_y_search_004.h5")
    assert not output_file.exists(), "Cannot overwrite output-file"

    with h5py.File(output_file, mode="x") as file:
        file.create_dataset("omegas", data=omega0s)
        file.attrs["bifurcation_beta"]=bif_beta
        file.attrs["bifucration_volume"]=bif_volume
        file.create_dataset("search_bounds", data=[lower_bound, upper_bound])

    for i, omega0 in enumerate(omega0s):
        print(f"{i+1}/{omega0s.shape[0]} :: omega0={np.linalg.norm(omega0):.2e}")
        time, state, volume = bisectionSearch(
            low=lower_bound,
            high=upper_bound,
            setParameter=partial(setSimParameters, beta=bif_beta),
            runFunc=partial(singleRun, omega0=omega0),
            search_depth=SEARCH_DEPTH,
            rng=RNG,
        )
        with h5py.File(output_file, mode="r+") as file:
            group = file.create_group(f"{i}")
            group.attrs["omega"] = omega0
            group.attrs["bounds"] = (lower_bound, upper_bound)
            group.attrs["description"] = "Time and state values for each search point. Last search point is the final result."
            group.create_dataset("time", data=time)
            group.create_dataset("state", data=state).attrs["state_variables"] = "v_x v_y v_z n_x n_y n_z omega_x omega_y omega_z"
            group.create_dataset("volume", data=volume)
