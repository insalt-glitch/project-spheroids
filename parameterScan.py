import numpy as np
import math
import dynamics
import c_dynamics
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from typing import Callable
from dynamics import SystemConstants as Constants
import h5py

SEED = 1
ACCURACY = 1e-13
MAX_INTEGRATION_TIME = 1000
GRID_SIZE = 25
VOLUME_RANGE = (1e-12, 1e-10)
ASPECT_RATIO_RANGE = [(0.2, 0.9), (1.2, 5)]

def singleRun(y0, t_max, const: Constants):
    return c_dynamics.solveDynamics(
        y0=y0,
        const=const,
        t_span=(0.0, t_max),
        rel_tol=1e-12,
        abs_tol=1e-12,
        event_type=2,
    )

def initialConditions(
    RNG: np.random.Generator,
    const: Constants,
):
    x0 = np.zeros(shape=(3,))
    v0 = RNG.normal(
        loc=const.W_approx,
        scale=0.01 * const.W_approx,
        size=(3,))
    theta = RNG.normal(
        loc=np.pi / 2,
        scale=0.01 * 2 * np.pi)
    n0 = np.stack([np.sin(theta), 0.0, np.cos(theta)])
    omega0 = RNG.normal(
        loc=0.0,
        scale=0.01 / const.tau_p,
        size=(3,))
    return np.concat([x0, v0, n0, omega0])

def searchForBifurcation(
    low: float,
    high: float,
    t_max: float,
    const: Constants,
    setParameter: Callable[[Constants, float], Constants],
    search_depth: int,
    RNG: np.random.Generator
):
    ts = np.empty(shape=search_depth, dtype=np.float64)
    ys = np.empty(shape=(search_depth, 12), dtype=np.float64)

    for i in range(search_depth):
        mid = (high + low) / 2
        print(f"    iter {i:02} :: ({low:.2e}, {high:.2e}) --> {mid:.2e}")
        const = setParameter(const, mid)
        y0 = initialConditions(RNG, const)
        t, y = singleRun(
            y0=y0,
            t_max=t_max, const=const)
        ts[i] = t
        ys[i] = y
        theta = math.acos(y[8])
        # print(theta * 180 / np.pi, t)
        assert not math.isnan(theta)
        if  abs(math.pi / 2 - theta) * 180 / math.pi > 1:
            high = mid
        else:
            low = mid
    return ts, ys, mid

def setVolume(const: Constants, volume: float):
    a_perp, a_para = dynamics.spheriodDimensionsFromBeta(const.beta, volume)
    new_parameters = Constants(
        a_para=a_para,
        a_perp=a_perp,
    )
    return new_parameters

def searchDepth(low, high):
    return math.ceil(math.log2(abs(high - low) / ACCURACY))

if __name__ == "__main__":
    RNG = np.random.default_rng(SEED)
    num_runs = GRID_SIZE
    # NOTE: The oscillation is quite unstable, which means we have start from a known "good" point.
    #     <interval-size> / 2^<search-depth> = <accuracy>
    # <=> <search-depth> = log2(<interval-size> / <accuracy>)

    const = Constants()
    betas = np.concat([
        np.linspace(
            start=span[0],
            stop=span[1],
            num=GRID_SIZE) for span in ASPECT_RATIO_RANGE
    ])

    # print(const.rho_p / const.rho_f)
    # t, y, volume = searchWrapper(0.2)
    # theta = np.arccos(np.array(y)[:,8])
    # import matplotlib.pyplot as plt
    # print(volume / dynamics.particleVolume(const.a_perp, const.a_para), dynamics.particleVolume(const.a_perp, const.a_para))
    # plt.plot(t[6:], curlyR[6:])
    # plt.show()
    with h5py.File("parameter_scan_001.h5", mode="w") as file:
        file.create_dataset("betas", data=betas)

        search_range = VOLUME_RANGE
        for i, beta in enumerate(betas):
            search_depth = searchDepth(search_range[0], search_range[1])
            print(f"{i+1}/{betas.size} :: beta={beta:.2f} | search-depth: {search_depth}")
            group = file.create_group(f"{i}")
            integration_times, final_states, particle_volume = searchForBifurcation(
                low=search_range[0],
                high=search_range[1],
                t_max=MAX_INTEGRATION_TIME,
                const=Constants(
                    a_para=const.a_para * beta,
                    a_perp=const.a_para
                ),
                setParameter=setVolume,
                search_depth=search_depth,
                RNG=RNG
            )
            group.create_dataset("integration_times", data=integration_times)
            group.create_dataset("final_states", data=final_states)
            group.create_dataset("particle_volume", data=particle_volume)

            search_range = [
                particle_volume * 0.5,
                particle_volume * 2.0
            ]
    # TODO: Found bifurcation up to idx 20 -> beta=0.78


