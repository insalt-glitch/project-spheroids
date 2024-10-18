import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
import dynamics
import c_dynamics
from pathlib import Path
from tqdm import tqdm

FOLDER_FIGURES = Path("figures")
STYLE_FILE = "plot_style.mplstyle"

class IntergateEvent:
    buf_size = 100
    buf = np.ones(buf_size)
    buf_idx = 0

    def reset():
        IntergateEvent.buf_idx = 0
        IntergateEvent.buf = np.ones(IntergateEvent.buf_size)

def detect_steady_state(t, y, args) -> float:
    IntergateEvent.buf[IntergateEvent.buf_idx] = np.linalg.norm(y[9:12])
    IntergateEvent.buf_idx = (IntergateEvent.buf_idx + 1) % IntergateEvent.buf_size
    return np.any(IntergateEvent.buf > 1e-5)

detect_steady_state.terminal = True

def plotCoefficientsVsReynoldsNumber(save=False):
    """Plot the coefficients C_F and C_T against the Reynolds number for testing
    """
    arr_beta = [0.2, 0.25, 0.5, 0.8, 1.25, 2, 4, 5]
    arr_Re_p = np.logspace(-1, 2, 1001)

    plt.style.use("plot_style.mplstyle")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    ax_cf, ax_ct = axes
    for beta in arr_beta:
        arr_C_F = np.zeros_like(arr_Re_p)
        arr_C_T = np.zeros_like(arr_Re_p)
        for i, Re_p in enumerate(arr_Re_p):
            shape_factor = dynamics.shapeFactor(beta)
            arr_C_F[i] = dynamics.correctionFactorStokesForce(Re_p, beta)
            arr_C_T[i] = dynamics.correctionFactorTorque(Re_p, beta, shape_factor)
        ax_cf.plot(arr_Re_p, arr_C_F, label=f"{beta:.2f}")
        ax_ct.plot(arr_Re_p, arr_C_T, label=f"{beta:.2f}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e2)
        ax.set_xlabel("Particle Reynolds number - Re$_p$")
        ax.legend(title="Aspect ratio $\\beta$")
    ax_cf.set_ylabel("Correction coefficient Stokes Force - C$_F$")
    ax_ct.set_ylabel("Correction coefficient Torque - C$_T$")
    if save:
        plt.savefig(FOLDER_FIGURES / "correction_coefficient-vs-reynolds_number.png", dpi=500, bbox_inches="tight")
    plt.show()

def plotSettlingSpeedVsAspectRatio(save=False):
    rng = np.random.default_rng(0)
    const = dynamics.SystemConstants()
    num_points = 20
    betas = np.hstack([
        np.logspace(np.log10(0.1), np.log10(0.9), num=num_points),
        np.logspace(np.log10(1.1), np.log10(6), num=num_points),
    ])
    v_g_arr = []
    for beta in tqdm(betas):
        n0 = rng.normal(size=3)
        n0 = n0 / np.linalg.norm(n0)
        a_perp, a_para = dynamics.spheriodDimensionsFromBeta(beta, const.particle_volume)
        const = dynamics.SystemConstants(a_perp=a_perp, a_para=a_para)
        c_config = c_dynamics.CppConfig(const)
        IntergateEvent.reset()
        solve_result = integrate.solve_ivp(
            fun=c_dynamics.systemDynamics,
            t_span=(0, 10.0,),
            y0=np.concat([[0, 0, 0], [0, 0, 1e-2], n0, [0, 0, 0]]),
            args=(c_config,),
            method="RK45",
            events=detect_steady_state,
        )
        if solve_result.status == 0:
            v = np.linalg.norm(solve_result.y[3:6], axis=0)
            v_g_arr.append(np.mean(v[-10:]))
            v_g_std = np.std(v[-10:]) / np.sqrt(10)
        else:
            print(f"Status ({solve_result.status}) :: {solve_result.message}")
            break
    # num_points = min(len(v_g_arr), num_points)
    plt.style.use("plot_style.mplstyle")
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    h = ax.plot(betas[:num_points], v_g_arr[:num_points])
    ax.plot(betas[num_points:], v_g_arr[num_points:], color=h[-1].get_color(), ls=h[-1].get_linestyle())
    ax.set(
        xlabel = "Aspect ratio $\\lambda$",
        ylabel = "Settling speed $v_g^*$ (m/s)",
        xlim = (0.15, 6),
        ylim = (0.2, 1.6),
        xscale = "log",
        xticks = [0.2, 0.5, 1, 2, 5],
        yticks = [0.4, 0.8, 1.2, 1.6],
    )
    # ax.tick_params(axis='x', which='minor')
    minor_locator = matplotlib.ticker.AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if save:
        plt.savefig(FOLDER_FIGURES / "settling_velocity-vs-aspect_ratio.png", dpi=500, bbox_inches="tight")
    plt.show()

def plotKPhiFormula(save=False):
    def KPhi0MATLAB(beta):
        gamma = np.log(beta + np.sqrt(beta ** 2 - 1 + 0j)) / (beta * np.sqrt(beta ** 2 - 1 + 0j))
        gamma = np.real(gamma)
        A_p = (4 * (beta ** 2 - 1)) / (3 * beta * ((2 * beta ** 2 - 1) * gamma - 1))
        K_phi0 = A_p / beta ** (1 / 3)
        return K_phi0

    def KPhi0Paper(beta):
        K_phi0 = (8 / 3) * beta ** (-1 / 3) / (
            + 2 * beta / (1 - beta ** 2)
            + 2 * (1 - 2 * beta ** 2) / (1 - beta ** 2) ** (3 / 2) * np.arctan(np.sqrt(1 - beta ** 2) / beta)
        )
        return K_phi0

    betas = np.linspace(0.01, 0.99, 99)
    k_phi_matlab = KPhi0MATLAB(betas)
    _, k_phi_paper = KPhi0Paper(betas)
    k_phi_paper /= betas ** (1 / 3)
    plt.figure(figsize=(6,4))
    plt.style.use("plot_style.mplstyle")
    plt.plot(betas, k_phi_matlab, label="MATLAB (l. 6745)")
    plt.plot(betas, k_phi_paper, label="Paper/MATLAB (l. 6062)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Aspect ratio $\\lambda$")
    plt.ylabel("Coefficient K$_{\\phi=0\\deg}$")
    plt.legend(title="Formulas")
    if save:
        plt.savefig(FOLDER_FIGURES / "K_phi=0-formula_comparison.png", dpi=500, bbox_inches="tight")
    plt.show()

def plotCorrectionCoefficients(save=False):
    const = dynamics.SystemConstants()
    plt.style.use("plot_style.mplstyle")
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    particle_volumes = [
        1.44e-3 * 1e-9, # m^3
        2.08e-3 * 1e-9, # m^3
        28.28e-3 * 1e-9, # m^3
    ]
    for i, volume in enumerate(particle_volumes):
        C_F_arrs = [[], []]
        C_T_arrs = [[], []]
        beta_arrs = [
            np.logspace(np.log10(0.1), np.log10(0.8)),
            np.logspace(np.log10(1.25), np.log10(6)),
        ]
        for j, betas in enumerate(beta_arrs):
            for beta in betas:
                a_perp, a_para = dynamics.spheriodDimensionsFromBeta(beta, volume)
                const = dynamics.SystemConstants(a_perp=a_perp, a_para=a_para)
                Re_p0 = dynamics.particleReynoldsNumber(const.a_perp, const.a_para, const.W_approx, const.nu)
                C_F = dynamics.correctionFactorStokesForce(Re_p0, const.beta, full_solve=False)
                v_g_star = dynamics.steadyStateSettlingSpeed(C_F, const.a_perp, const.tau_p, const.A_g, const.nu, const.g)
                Re_p = dynamics.particleReynoldsNumber(const.a_perp, const.a_para, v_g_star, const.nu)
                C_T = dynamics.correctionFactorTorque(Re_p, const.beta, const.F_lambda)
                C_F_arrs[j].append(C_F)
                C_T_arrs[j].append(C_T)


        for ax, coeff_arrs in zip(axes, [C_F_arrs, C_T_arrs]):
            i_roman = 'I'*(i+1)
            h = ax.plot(beta_arrs[0], coeff_arrs[0], label=f"Group {i_roman}")
            ax.plot(beta_arrs[1], coeff_arrs[1], color=h[-1].get_color(), ls=h[-1].get_linestyle())

    for ax, coeff_name in zip(axes, ["C$_F$", "C$_T$"]):
        ax.set(
            xlim = (0.15, 6),
            ylim = (0, 1),
            xscale = "log",
            xticks = [0.2, 0.5, 1.0, 2.0, 5.0],
            yticks = [0, 0.5, 1],
            xlabel = "Aspect ratio $\\lambda$",
            ylabel = f"Coefficient {coeff_name}",
        )
        ax.tick_params(axis='x', which='minor')
        minor_locator = matplotlib.ticker.AutoMinorLocator(5)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(FOLDER_FIGURES / "correction_coefficients-C_F-C_T-vs-aspect_ratio.png", dpi=500, bbox_inches="tight")
    plt.show()

def plotShapeFactor(save=False):
    betas = np.logspace(-2, 2, num=200)
    F_lambdas = [dynamics.shapeFactor(beta) for beta in betas]
    plt.style.use("plot_style.mplstyle")
    plt.figure(figsize=(6,4))
    plt.plot(betas, F_lambdas)
    plt.xscale("log")
    plt.xlabel("Aspect ratio $\\lambda$")
    plt.ylabel("Shape factor $F(\\lambda)$")
    plt.xlim(1e-2, 1e2)
    plt.ylim(-1.5, 2.75)
    if save:
        plt.savefig(FOLDER_FIGURES / "shape_factor-vs-aspect_ratio.png", dpi=500, bbox_inches="tight")
    plt.show()

def plotLowGravitySettlingSpeed(save=False):
    const = dynamics.SystemConstants()
    x = const.nu / (const.tau_p * const.a_perp)
    gravities = np.linspace(0.01, 0.02) * x
    v_g_arr = []
    W_arr = []
    for g in gravities:
        const = dynamics.SystemConstants(gravitational_acceleration=g)
        solve_result = integrate.solve_ivp(
            fun=dynamics.systemDynamics,
            t_span=(0, 2.0,),
            y0=np.concat([[0, 0, 0], [0, 0, 1e-2], [0, 1, 0], [0, 0, 0]]),
            args=(const,),
            method="RK45",
        )
        print(f"beta == {const.beta:.2f} | Steps == {solve_result.t.size}")
        if solve_result.status == 0:
            v = np.linalg.norm(solve_result.y[3:6], axis=0)
            v_g = np.mean(v[-10:])
            v_g_std = np.std(v[-10:]) / np.sqrt(10)
            print(f"settling velocity == ({v_g:.2f}+-{v_g_std:.1e})m/s")
            v_g_arr.append(v_g)
            W_arr.append(const.W_approx)
        else:
            print(f"Status ({solve_result.status}) :: {solve_result.message}")
            break
    plt.style.use("plot_style.mplstyle")
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    h = ax.plot(gravities / x, v_g_arr, label="Simulation")
    ax.plot(gravities / x, W_arr, label="Settling speed $W=g\\tau_p/A^{(g)}$")
    ax.set(
        xlabel = "Normalized gravity $ga_\\perp\\tau_p/\\nu$",
        ylabel = "Settling speed $v_g^*$ (m/s)",
    )
    ax.legend()
    if save:
        plt.savefig(FOLDER_FIGURES / "settling_speed-vs-low_gravity.png", dpi=500, bbox_inches="tight")
    plt.show()

def discriminantDelta(const: dynamics.SystemConstants):
    v_scale = np.linalg.norm(const.g) * const.tau_p / const.A_g
    vg_nondim = const.W_approx / v_scale
    Re_p0 = dynamics.particleReynoldsNumber(const.a_perp, const.a_para, const.W_approx, const.nu)
    C_T = dynamics.correctionFactorTorque(Re_p0, const)
    curly_R = const.rho_p / const.rho_f
    curly_V = np.linalg.norm(const.g) * const.particle_volume / const.nu ** 2
    delta = 1 - 4 * vg_nondim ** 2 * C_T * curly_R ** 3 * curly_V ** 2

if __name__ == '__main__':
    # plotSettlingSpeedVsAspectRatio()
    fac = 1 # 4
    const = dynamics.SystemConstants()
    const = dynamics.SystemConstants(a_para=const.a_para * fac, a_perp=const.a_perp * fac)

    x0 = [0.1, 0.2, 0.3]
    v0 = [5e-2, 7e-2, 3e-2]
    n0 = np.array([1, 2, 3]) / np.linalg.norm([1,2,3])
    omega0 = [0.13, 0.2, 0.78]
    y0 = np.concat([x0, v0, n0, omega0])

    t = np.linspace(0.0, 50.0, num=10_000)
    t, res = c_dynamics.solveDynamics(
        y0, t,
        const,
        rel_tol=1e-12,
        abs_tol=1e-12,
        event_type=1,
    )
    res = res.T
    n = res[6:9]
    phi = np.arccos(n[2])
    theta = np.sign(n[1]) * np.arccos(n[0] / np.linalg.norm(n[:2], axis=0))
    plt.style.use(STYLE_FILE)
    plt.plot(t, phi * 180 / np.pi, label="$\\varphi(t)$")
    plt.plot(t, theta * 180 / np.pi, label="$\\theta(t)$")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.yticks([-30, 0, 30, 60, 90, 120])
    plt.ylim(-30, 120)
    plt.legend()
    plt.tight_layout()
    plt.show()
