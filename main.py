import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
import dynamics
import c_dynamics
from pathlib import Path
from tqdm import tqdm
import ctypes

FOLDER_FIGURES = Path("figures")
STYLE_FILE = Path("plot_style.mplstyle")
FIGURE_FORMAT = "svg"
FIGURE_DPI = 300

def saveFigure(figure_name: str) -> None:
    folder = FOLDER_FIGURES / FIGURE_FORMAT
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / f"{figure_name}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")

def plotCoefficientsVsReynoldsNumber(save=False):
    """Plot the coefficients C_F and C_T against the Reynolds number for testing
    """
    arr_beta = [0.2, 0.25, 0.5, 0.8, 1.25, 2, 4, 5]
    arr_Re_p = np.logspace(-1, 2, 1001)

    plt.style.use(STYLE_FILE)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    ax_cf, ax_ct = axes
    for beta in arr_beta:
        arr_C_F = np.zeros_like(arr_Re_p)
        arr_C_T = np.zeros_like(arr_Re_p)
        for i, Re_p in enumerate(arr_Re_p):
            const = dynamics.SystemConstants(a_para=beta, a_perp=1.0)
            c_config = c_dynamics.CppConfig(const)
            arr_C_F[i] = c_config.correctionFactorStokesForce(Re_p)
            arr_C_T[i] = c_config.correctionFactorTorque(Re_p)
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
        saveFigure("correction_coefficient-vs-reynolds_number")
    plt.show()

def plotSettlingSpeedVsAspectRatio(save=False):
    rng = np.random.default_rng(0)
    num_points = 10
    num_repetitions = 2
    t_eval = np.concat([[0.0], np.linspace(19.0, 20.0)])
    betas = np.hstack([
        np.logspace(np.log10(0.1), np.log10(0.9), num=num_points),
        np.logspace(np.log10(1.1), np.log10(6), num=num_points),
    ])
    particle_volumes = [
        1.44e-3 * 1e-9, # m^3
        2.08e-3 * 1e-9, # m^3
        28.28e-3 * 1e-9, # m^3
    ]
    markers = ["o", "p", "v"]
    colors = ['#e69f00', '#56b4e9', '#009e73']
    plt.style.use(STYLE_FILE)
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    for volume_idx, (volume, marker, color) in enumerate(zip(particle_volumes, markers, colors)):
        v_settle = np.empty(betas.size)
        v_settle_err = np.empty(betas.size)
        for beta_idx, beta in enumerate(tqdm(betas)):
            a_perp, a_para = dynamics.spheriodDimensionsFromBeta(beta, volume)
            const = dynamics.SystemConstants(a_perp=a_perp, a_para=a_para)
            c_config = c_dynamics.CppConfig(const)
            v_reps = np.empty(num_repetitions)
            for rep_idx in range(num_repetitions):
                # initial conditions
                x0 = np.array([0.0, 0.0, 0.0])
                n0 = rng.normal(size=3)
                n0 = n0 / np.linalg.norm(n0)
                v0 = rng.normal(size=3) * 0.1
                omega0 = rng.normal(size=3) * 0.01
                y0 = np.concat([x0, v0, n0, omega0])
                # run simulation
                t, result = c_dynamics.solveDynamics(
                    y0=y0, const=const, t_eval=t_eval,
                    rel_tol=1e-12, abs_tol=1e-12
                )
                v_reps[rep_idx] = np.mean(np.linalg.norm(result[1:,3:6], axis=1))
            v_settle[beta_idx] = np.mean(v_reps) * const.g * const.tau_p
            v_settle_err[beta_idx] = np.std(v_reps) / np.sqrt(t_eval.size - 2) * const.g * const.tau_p
        # plotting
        i_roman = 'I' * (volume_idx + 1)
        h = ax.errorbar(betas[:num_points], v_settle[:num_points], yerr=v_settle_err[:num_points], ls=":", color=color, marker=marker, markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax.errorbar(betas[num_points:], v_settle[num_points:], yerr=v_settle_err[num_points:], ls=":", marker=marker, markersize=8, markeredgecolor="black", markeredgewidth=1, color=color, label=f"Group {i_roman}")
    ax.set(
        xlabel = "Aspect ratio $\\lambda$",
        ylabel = "Settling speed $v_g^*$ (m/s)",
        xlim = (0.15, 6),
        ylim = (0.2, 1.6),
        xscale = "log",
        xticks = [0.2, 0.5, 1, 2, 5],
        yticks = [0.4, 0.8, 1.2, 1.6],
    )
    minor_locator = matplotlib.ticker.AutoMinorLocator(4)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc="center right")
    if save:
        saveFigure("settling_velocity-vs-aspect_ratio")
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
    plt.style.use(STYLE_FILE)
    plt.plot(betas, k_phi_matlab, label="MATLAB (l. 6745)")
    plt.plot(betas, k_phi_paper, label="Paper/MATLAB (l. 6062)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Aspect ratio $\\lambda$")
    plt.ylabel("Coefficient K$_{\\phi=0\\deg}$")
    plt.legend(title="Formulas")
    if save:
        saveFigure("K_phi=0-formula_comparison")
    plt.show()

def plotCorrectionCoefficients(save=False):
    const = dynamics.SystemConstants()
    plt.style.use(STYLE_FILE)
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
                c_config = c_dynamics.CppConfig(const)
                C_F = c_config.correctionFactorStokesForce(const.Re_p0)
                v_g_star = dynamics.steadyStateSettlingSpeed(C_F, const.a_perp, const.tau_p, const.A_g, const.nu, const.g)
                Re_p = dynamics.particleReynoldsNumber(const.a_max, v_g_star, const.nu)
                C_T = c_config.correctionFactorTorque(Re_p)

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
        saveFigure("correction_coefficients-C_F-C_T-vs-aspect_ratio")
    plt.show()

def plotShapeFactor(save=False):
    betas = np.logspace(-2, 2, num=200)
    F_lambdas = [dynamics.shapeFactor(beta) for beta in betas]
    plt.style.use(STYLE_FILE)
    plt.figure(figsize=(6,4))
    plt.plot(betas, F_lambdas, color="black")
    plt.xscale("log")
    plt.xlabel("Aspect ratio $\\lambda$")
    plt.ylabel("Shape factor $F(\\lambda)$")
    plt.xlim(1e-2, 1e2)
    plt.ylim(-1.5, 2.75)
    if save:
        saveFigure("shape_factor-vs-aspect_ratio")
    plt.show()

def discriminantDelta(const: dynamics.SystemConstants):
    v_scale = np.linalg.norm(const.g) * const.tau_p / const.A_g
    vg_nondim = const.W / v_scale
    c_config = c_dynamics.CppConfig(const)
    C_T = c_config.correctionFactorTorque(const.Re_p0)
    delta = 1 - 4 * vg_nondim ** 2 * C_T * const.curly_R ** 3 * const.curly_V ** 2

def checkCoefficents():
    folder = Path("coefficients_full")
    for file in folder.glob("*.txt"):
        beta = float(file.stem.split("lamRod")[-1].replace("_", "."))
        print(f"aspect-ratio = {beta:.2f}")
        data = np.loadtxt(file)
        Re = data[:,0]
        C_F_ref = data[:,1]
        C_T_ref = data[:,2]
        const = dynamics.SystemConstants(a_para=beta, a_perp=1)
        config = c_dynamics.CppConfig(const)

        config.c_dll.correctionFactorStokesForce.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(c_dynamics.CppConstantsStruct)
        ]
        config.c_dll.correctionFactorTorque.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(c_dynamics.CppConstantsStruct)
        ]
        config.c_dll.correctionFactorStokesForce.restype = ctypes.c_double
        config.c_dll.correctionFactorTorque.restype = ctypes.c_double
        C_F_cpp = [config.c_dll.correctionFactorStokesForce(reynold, ctypes.byref(config.cpp_constants_struct)) for reynold in Re]
        C_T_cpp = [config.c_dll.correctionFactorTorque(reynold, ctypes.byref(config.cpp_constants_struct)) for reynold in Re]
        print(f"C_F diff :: mean = {np.mean(np.abs(C_F_cpp - C_F_ref)):.4e} | max = {np.max(np.abs(C_F_cpp - C_F_ref)):.4e}")
        print(f"C_T diff :: mean = {np.mean(np.abs(C_T_cpp - C_T_ref)):.4e} | max = {np.max(np.abs(C_T_cpp - C_T_ref)):.4e}")
        print()

if __name__ == '__main__':
    print("Nothing here")
    # # detect sign change omega > 0 -> omega < 0 and record theta. Then detect theta = constant with rolling buffer
    # # do binary search to find bifurcation point for a given set of parameters.
    # # search in curlyR curlyV and lambda space.
    # # lambda -> aspect ratio
    # # curlyR -> density ratio
    # # curlyV -> particle volume
    # # plotSettlingSpeedVsAspectRatio()
    # # 4 -> osciallation
    # fac = 2 # 3.9375
    # a_perp, a_para = dynamics.spheriodDimensionsFromBeta(0.2,  9.2e-11)
    # const = dynamics.SystemConstants(a_para=a_para, a_perp=a_perp)
    # # const = dynamics.SystemConstants(a_para=const.a_para * fac, a_perp=const.a_perp * fac)

    # RNG = np.random.default_rng(0)
    # # for i in range(10):
    # x0 = np.zeros((3,))
    # v0 = RNG.normal(size=(3,)) # 1%
    # n0 = RNG.normal(size=(3,)) # 0.1%
    # n0 /= np.linalg.norm(n0, keepdims=True)
    # omega0 = RNG.normal(size=(3,)) # 0.01 / tau_p
    # y0 = np.concat([x0, v0, n0, omega0])

    # t = np.linspace(0, 10, num=1_000)
    # t, res = c_dynamics.solveDynamics(
    #     y0=y0,
    #     const=const,
    #     t_eval=t,
    #     t_span=(0.0, 10.0),
    #     rel_tol=1e-12,
    #     abs_tol=1e-12,
    #     event_type=0
    # )

    # res = res.T
    # n = res[6:9]
    # theta = np.arccos(n[2])
    # # print(t, theta * 180 / np.pi)
    # # print(f"t: {t} | theta: {theta * 180 / np.pi}")
    # phi = np.sign(n[1]) * np.arccos(n[0] / np.linalg.norm(n[:2], axis=0))
    # plt.style.use(STYLE_FILE)
    # # plt.plot(t[t>40], phi[t>40] * 180 / np.pi, label="$\\varphi(t)$")
    # plt.plot(t, theta * 180 / np.pi, label="$\\theta(t)$")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angle (deg)")
    # plt.yticks([-30, 0, 30, 60, 90, 120])
    # plt.ylim(-30, 120)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
