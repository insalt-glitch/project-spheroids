import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple
from scipy import optimize

class SystemConstants:
    def __init__(
        self,
        a_para: float = 47.9e-6  / 2,
        a_perp: float = 239.4e-6 / 2,
        particle_density: float = 1.2e3,
        fluid_density: float = 1.204,
        fluid_kinematic_viscosity: float = 1.51147e-5,
        gravitational_acceleration: float = 9.81,
    ):
        """Collection of constants required to solve the differential equations.

        Basic particle properties:
            a_perp   (float): particle radius (perpendicular to symmetry axis)
            a_para   (float): particle radius (parallel      to symmetry axis)
            beta     (float): aspect ratio
            nu       (float): kinematic viscosity of the surrounding fluid/gas
            m_p      (float): particle mass
            tau_p    (float): particle response time
            W_approx (float): approximate particle settling speed
            F_beta   (float): particle shape factor
            g   (np.ndarray): gravitational acceleration
        Correction factors:
            C_F      (float): Stokes force (quiescent fluid)
            C_T      (float): torque       (quiescent fluid)
        Particle interia tensor:
            I_perp   (float): coefficient (perpendicular to symmetry axis)
            I_para   (float): coefficient (parallel      to symmetry axis)
        Translation resistance tensor:
            A_perp   (float): coefficient (perpendicular to symmetry axis)
            A_para   (float): coefficient (parallel      to symmetry axis)
            A_g      (float): component in direction of gravity (in steady state)
        Rotation resistance tensor:
            C_perp   (float): coefficient (perpendicular to symmetry axis)
            C_para   (float): coefficient (parallel      to symmetry axis)

        Args:
            a_para (float, optional): particle radius (parallel      to
                symmetry axis). Defaults to 47.9e-6/2.
            a_perp (float, optional): particle radius (perpendicular to
                symmetry axis). Defaults to 239.4e-6/2.
            particle_density (float, optional): density of the particle. Defaults to 1.2e3.
            fluid_density (float, optional): mass density of the
                surrounding fluid/gas. Defaults to 1.204.
            fluid_kinematic_viscosity (float, optional): kinematic viscosity
                of the surrounding fluid/gas. Defaults to 1.51147e-5.
            gravitational_acceleration (float, optional): gravitational
                acceleration. Defaults to 9.81.
        """
        # parameters
        self.g = gravitational_acceleration
        self.nu = fluid_kinematic_viscosity

        self.a_perp = a_perp
        self.a_para = a_para
        self.beta = aspectRatio(a_perp, a_para)
        self.a_max = a_perp * max(1.0, self.beta)
        self.particle_volume = 4 / 3 * np.pi * a_perp ** 3 * self.beta
        self.F_beta = shapeFactor(self.beta)

        self.curly_R = particle_density / fluid_density
        self.curly_V = self.g * self.particle_volume / self.nu ** 2
        self.tau_p = particleResponseTime(
            a_perp,
            a_para,
            particle_density,
            fluid_density,
            fluid_kinematic_viscosity,
        )
        # quantites for the C program
        self.A_g = resistanceCoefficient(self.beta)
        self.W = approximateSettlingSpeed(self.beta, self.tau_p, self.g)

        self.A_perp, self.A_para = translationalResistanceCoefficients(self.beta)
        self.C_perp, self.C_para = rotationalResistanceCoefficients(self.beta)
        self.J_perp, self.J_para = particleInteriaTensorCoefficents(self.beta)

        self.Re_p0 = particleReynoldsNumber(self.a_max, self.W, self.nu)
        self.curly_A_F = self.curly_R * self.curly_V / (32 * np.pi)
        self.curly_A_T = self.beta * max(1, self.beta) ** 3 * self.F_beta * self.curly_R ** 3 * self.curly_V ** 2 / (972 * np.pi ** 3)

        # ----------------- Constans for precomputation -----------------
        self._fac_Re_p = self.a_max / self.nu
        self._fac1_v_g_star = 4 * self.nu / (3 * a_perp * self.A_g)
        self._fac2_v_g_star = 3 * a_perp * self.g * self.tau_p / (2 * self.nu)
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Empirical correlation coefficients (Page 32 - Table 5 c_{t,i})
        self._c_t = [0.931, 0.675, 0.162, 0.657, 2.77, 0.178, 0.177]
        self._C_T_prolate_c0 = self._c_t[5] + self._c_t[6] * np.log(self.beta)
        self._C_T_prolate_c1 = np.real(
            self._c_t[0] * np.log(self.beta + 0j) ** self._c_t[1]
            * (self.beta ** (2 / 3) / 2) ** self._c_t[2] * np.pi
            / (self.beta ** 2 * np.abs(self.F_beta))
        )
        self._C_T_prolate_c2 = np.real(
            self._c_t[3] * np.log(self.beta + 0j) ** self._c_t[4]
            * (self.beta ** (2 / 3) / 2) ** self._C_T_prolate_c0 * np.pi
            / (self.beta ** 2 * np.abs(self.F_beta))
        )
        self._C_T_oblate_c0 = np.real(
            np.pi * self.beta / (max(1, self.beta) ** 3) * 1.85
            * ((1 - self.beta + 0j) / self.beta) ** 0.832 * (max(1, self.beta)
            / (2 * self.beta ** (1 / 3))) ** 0.146
            / (2 * np.abs(self.F_beta))
        )

def spheriodDimensionsFromBeta(beta: float, particle_volume: float) -> Tuple[float, float]:
    # beta = a_para / a_perp
    #     V = 4 / 3 * np.pi * a_perp ** 2 * a_para
    # <=> V = 4 / 3 * np.pi * a_perp ** 3 * beta
    # <=> a_perp = V / (4 / 3 * np.pi * beta) ** (1 / 3)
    a_perp = (particle_volume / (4 / 3 * np.pi * beta)) ** (1 / 3)
    a_para = beta * a_perp
    return a_perp, a_para

def particleVolume(a_perp: float, a_para: float) -> float:
    """Calculate volume of the particle from its radii along and perpendicular
    to its symmetry axis.

    Formula taken from here: https://en.wikipedia.org/wiki/Spheroid
    Variable names (here <-> Wikipedia <-> Paper):
        a_perp <-> a <-> a_perpenticular
        a_para <-> c <-> a_parallel

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)

    Returns:
        float: the volume of the particle
    """
    return 4 / 3 * np.pi * a_perp ** 2 * a_para

def particleMass(particle_volume: float, rho_p: float) -> float:
    """Calculate the mass m_p of a spheroidal particle given its volume and density

    Args:
        particle_volume (float): spheriod volume
        rho_p (float): density of the particle

    Returns:
        float: mass of the particle
    """
    return particle_volume * rho_p

def resistanceCoefficient(beta: float) -> float:
    """Calculate the resistance coefficient A^(g). It is the component of the
    translation resistance tensor in direction of gravity (in steady state).

    Args:
        beta (float): ratio of the radii of a spheroid (aspect ratio)

    Returns:
        float: A^(g) component of the translation resistance tensor in direction
            of gravity (in steady state)
    """
    A_perp, A_para = translationalResistanceCoefficients(beta)
    return A_perp if (beta > 1) else A_para

def particleResponseTime(
    a_perp: float,
    a_para: float,
    rho_p: float,
    rho_f: float,
    nu: float,
) -> float:
    """Calculate particle response time tau_p.

    Formula taken from here:
        https://doi.org/10.1103/PhysRevLett.132.034101 (Page 5 - Appendix A)
    There the particle response time has the variable tau_p

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)
        rho_p (float): density of the particle
        rho_f (float): density of the fluid/gas around the particle
        nu (float): kinematic viscosity of the surrounding fluid/gas

    Returns:
        float: particle response time tau_p
    """
    return 2 * a_para * a_perp * rho_p / (9 * rho_f * nu)

def aspectRatio(a_perp: float, a_para: float) -> float:
    """Calculate the aspect ratio beta of a spheriodal particle.

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)

    Returns:
        float: aspect ratio
    """
    return a_para / a_perp

def approximateSettlingSpeed(
    beta: float,
    tau_p: float,
    g: np.ndarray|float,
) -> float:
    """Aproximate the slip velocity W of the particle at the fixed point (equilibrium/settingling velocity).

    Formula taken from here: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of resistance coefficient A_g and slip velocity W in the paper as A^(g)
        (Supplementary Material 1 - Empirical parameters - Page 7)

    Args:
        beta (float): aspect-ratio
        tau_p (float): particle response time
        g (np.ndarray|float): gravitational acceleration

    Returns:
        float: slip velocity
    """
    A_g = resistanceCoefficient(beta)
    return np.linalg.norm(g) * tau_p / A_g

def steadyStateSettlingSpeed(
    C_F: float,
    a_perp: float,
    tau_p: float,
    A_g: float,
    nu: float,
    g: np.ndarray|float,
) -> float:
    """Compute the steady-state settling speed (v_g^*) of the particle

    Formula taken from here: https://doi.org/10.1103/PhysRevLett.132.034101
        (Page 6 - Appendix A - Eq. A4)

    Args:
        C_F (float): correction coefficient for the Stokes force
        a_perp (float): particle radius (perpendicular to symmetry axis)
        tau_p (float): particle response time
        A_g (float): component in direction of gravity (in steady state)
        nu (float): kinematic viscosity of the surrounding fluid/gas
        g (np.ndarray|float): gravitational acceleration

    Returns:
        float: steady-state settling speed
    """
    v_g_star = 4 * nu * (
        np.sqrt(1 + 3 * C_F * a_perp * np.linalg.norm(g) * tau_p / (2 * nu)) - 1
    ) / (3 * a_perp * A_g * C_F)
    return v_g_star

def particleReynoldsNumber(
    a_max: float,
    W: float,
    nu: float,
) -> float:
    """Calculate particle Reynolds number Re_p.

    Formula taken from here: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of leading order reynolds number Re_p (Page 5 - Eq. 11)

    Args:
        a_max (float): max(a_para, a_perp)
        W (float): (instantaneous) slip velocity
        nu (float): kinematic viscosity of the surrounding fluid/gas

    Returns:
        float: zeroth order particle Reynolds number
    """
    return  W * a_max / nu

def shapeFactor(beta: float) -> float:
    """Calculate the shape factor/function F_beta for the interial torque of spheriods.

    Formulas taken from here: https://doi.org/10.1017/jfm.2015.360
        (Page 149 - Chapter 4 - Eqs. 4.1 & 4.2)
    Variable names:
        F = shape factor
        e = spheriod eccentricity

    Args:
        beta (float): ratio of the radii of a spheroid (aspect ratio)

    Returns:
        float: shape factor
    """
    if beta in [1, np.inf]:
        return 0
    if beta == 0:
        return 2.376
    if beta > 1: # beta > 1 -> prolate spheroid
        e = np.sqrt(1 - 1 / beta ** 2)
        F = np.pi * e ** 2 * ( # checked for correctness :)
            - (420 * e + 2240 * e ** 3 + 4249 * e ** 5 - 2152 * e ** 7)
            + (420 + 3360 * e ** 2 + 1890 * e ** 4 - 1470 * e ** 6) * np.arctanh(e)
            - (1260 * e - 1995 * e ** 3 + 2730 * e ** 5 - 1995 * e ** 7) * np.arctanh(e) ** 2
        ) / (315 * ((e ** 2 + 1) * np.arctanh(e) - e) ** 2 * ((1 - 3 * e ** 2) * np.arctanh(e) - e))
    else: # beta < 1 -> oblate spheroid
        e = np.sqrt(1 - beta ** 2)
        fac = np.sqrt(1 - e ** 2)
        F = np.pi * e ** 2 * ( # checked for correctness :)
            + e * (-420 + 3500 * e ** 2 - 9989 * e ** 4 + 4757 * e ** 6)
            + 210 * (2 - 24 * e ** 2 + 69 * e ** 4 - 67 * e ** 6 + 20 * e ** 8) * np.arcsin(e) / fac
            + 105 * e * (12 - 17 * e ** 2 + 24 * e ** 4) * np.arcsin(e) ** 2
        ) / (315 * (-e * fac + (1 + 2 * e ** 2) * np.arcsin(e)) * (e * fac + (2 * e ** 2 - 1) * np.arcsin(e)) ** 2)
    return F

def translationalResistanceCoefficients(beta: float) -> Tuple[float, float]:
    """Calculate coefficient of the translational resistance tensor.

    Equations from: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of A_perp & A_para & gamma (Supplementary Material 1 - Page 1 - Eq. S1)

    Args:
        beta (float): ratio of the radii of a spheroid  (aspect ratio)

    Returns:
        Tuple[float, float]: coefficients in (perpendicular, parallel) direction
    """
    gamma = np.log(beta + np.sqrt(beta ** 2 - 1 + 0j)) / (beta * np.sqrt(beta ** 2 - 1 + 0j))
    assert abs(np.imag(gamma)) < 1e-10, "What happend?"
    gamma = np.real(gamma)
    A_perp = 8 * (beta ** 2 - 1) / (3 * beta * ((2 * beta ** 2 - 3) * gamma + 1))
    A_para = 4 * (beta ** 2 - 1) / (3 * beta * ((2 * beta ** 2 - 1) * gamma - 1))
    return A_perp, A_para

def rotationalResistanceCoefficients(beta: float) -> Tuple[float, float]:
    """Calculate coefficient of the rotational resistance tensor.

    Equations from: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of gamma (Supplementary Material 1 - Page 1 - Eq. S1)
    Definition of C_perp & C_para (Supplementary Material 1 - Page 1 - Eq. S4)

    Args:
        beta (float): aspect-ratio

    Returns:
        Tuple[float, float]: coefficients in (perpendicular, parallel) direction
    """
    gamma = np.log(beta + np.sqrt(beta ** 2 - 1 + 0j)) / (beta * np.sqrt(beta ** 2 - 1 + 0j))
    gamma = np.real(gamma)
    C_perp = + 8 * beta * (beta ** 4 - 1) \
            / (9 * beta ** 2 * ((2 * beta ** 2 - 1) * gamma - 1))
    C_para = - 8 * beta * (beta ** 2 - 1) \
        / (9 * (gamma - 1) * beta ** 2)
    return C_perp, C_para

def particleInteriaTensorCoefficents(
    beta: float
) -> Tuple[float, float]:
    """Calculate unique matrix elements in of the particle interia tensor
    (in particle coordinate frame)

    Equations from: https://doi.org/10.1175/JAS-D-20-0221.1
    Definitions of I_para & I_perp (Supplementary Material 1 - Page 1 - Eq. S5)
    In idmensionless units ( 1 / (m_p * a_perp^2 )

    Args:
        beta (float): aspect-ratio

    Returns:
        Tuple[float, float]: coefficients in the particle interia tensor
    """
    J_perp = (1 / 5) * (1 + beta ** 2)
    J_para = (2 / 5)
    return J_perp, J_para
