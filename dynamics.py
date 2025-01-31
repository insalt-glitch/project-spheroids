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
            F_lambda (float): particle shape factor
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
        self.rho_p = particle_density
        self.rho_f = fluid_density
        self.nu = fluid_kinematic_viscosity
        self.beta = aspectRatio(a_perp, a_para)
        self.particle_volume = particleVolume(a_perp, a_para)
        self.a_perp = a_perp
        self.a_para = a_para
        self.a = max(a_perp, a_para)
        self.F_lambda = shapeFactor(self.beta)
        self.g = np.array([gravitational_acceleration, 0, 0])
        self.m_p = particleMass(self.particle_volume, particle_density)
        self.tau_p = particleResponseTime(
            a_perp,
            a_para,
            particle_density,
            fluid_density,
            fluid_kinematic_viscosity,
        )
        self.W_approx = approximateSettlingSpeed(
            a_perp,
            a_para,
            particle_density,
            fluid_density,
            fluid_kinematic_viscosity,
            self.g,
        )
        self.C_perp, self.C_para = rotationalResistanceCoefficients(a_perp, a_para)
        self.A_perp, self.A_para = translationalResistanceCoefficients(self.beta)
        self.A_g = resistanceCoefficient(self.beta, self.A_perp, self.A_para)
        self.J_perp, self.J_para = particleInteriaTensorCoefficents(
            a_perp,
            a_para,
            self.m_p,
        )
        # derived constants for precomputation
        self._A_diff = self.A_para - self.A_perp
        self._C_diff = self.C_para - self.C_perp
        self._fac_TF_h0 = - (self.m_p / self.tau_p)
        self._fac_F_h1 = - (3 / 16) * (self.m_p / self.tau_p) * (self.a_perp / self.nu)
        self._fac_T_h1 = self.F_lambda * (self.m_p / (6 * np.pi)) \
            * (self.a ** 3 / (self.a_perp * self.nu)) / self.tau_p
        self._J_diff = self.J_perp - self.J_para
        self._fac_Re_p0 = self.a / self.nu
        self._fac1_v_g_star = 4 * self.nu / (3 * self.a_perp * self.A_g)
        self._fac2_v_g_star = 3 * self.a_perp * np.linalg.norm(self.g) * self.tau_p / (2 * self.nu)
        self._C_F_oblate_c0 = 0.7311124212687891 * self.beta ** 96.47233333333332
        self._C_F_oblate_c1 = np.real(1.453237823359436 * (1 - self.beta + 0j) ** 0.4374 * self.beta ** 0.5837333333333332)
        # Empirical correlation coefficients (Page 32 - Table 5 c_{d,i}) Fröhlich JFM 901 (2020)
        self._c_d = [-0.007, 1.0, 1.17, -0.07, 0.047, 1.14, 0.7, -0.008]
        self._C_F_prolate_c0 = self._c_d[6] + self._c_d[7] * np.log(self.beta)
        self._C_F_prolate_c1 = np.real(2 * 0.18277810531719724 * (self.beta + 0j) ** 0.229)
        self._C_F_prolate_c2 = np.real(
            2 * 0.75 ** (-self._C_F_prolate_c0) * self._c_d[4]
            * self.beta ** (self._C_F_prolate_c0 / 3)
            * np.log(self.beta + 0j) ** self._c_d[5]
        )
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Empirical correlation coefficients (Page 32 - Table 5 c_{t,i})
        self._c_t = [0.931, 0.675, 0.162, 0.657, 2.77, 0.178, 0.177]
        self._C_T_prolate_c0 = self._c_t[5] + self._c_t[6] * np.log(self.beta)
        self._C_T_prolate_c1 = np.real(
            self._c_t[0] * np.log(self.beta + 0j) ** self._c_t[1]
            * (self.beta ** (2 / 3) / 2) ** self._c_t[2] * np.pi
            / (self.beta ** 2 * np.abs(self.F_lambda))
        )
        self._C_T_prolate_c2 = np.real(
            self._c_t[3] * np.log(self.beta + 0j) ** self._c_t[4]
            * (self.beta ** (2 / 3) / 2) ** self._C_T_prolate_c0 * np.pi
            / (self.beta ** 2 * np.abs(self.F_lambda))
        )
        self._C_T_oblate_c0 = np.real(
            np.pi * self.beta / (max(1, self.beta) ** 3) * 1.85
            * ((1 - self.beta + 0j) / self.beta) ** 0.832 * (max(1, self.beta)
            / (2 * self.beta ** (1 / 3))) ** 0.146
            / (2 * np.abs(self.F_lambda))
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

def resistanceCoefficient(beta: float, A_perp: float, A_para: float ) -> float:
    """Calculate the resistance coefficient A^(g). It is the component of the
    translation resistance tensor in direction of gravity (in steady state).

    Args:
        beta (float): ratio of the radii of a spheroid (aspect ratio)
        A_perp (float): coefficient (perpendicular to symmetry axis) of the
            translation resistance tensor
        A_para (float): coefficient (parallel      to symmetry axis) of the
            translation resistance tensor

    Returns:
        float: A^(g) component of the translation resistance tensor in direction
            of gravity (in steady state)
    """
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
    a_perp: float,
    a_para: float,
    rho_p: float,
    rho_f: float,
    nu: float,
    g: np.ndarray|float,
) -> float:
    """Aproximate the slip velocity W of the particle at the fixed point (equilibrium/settingling velocity).

    Formula taken from here: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of resistance coefficient A_g and slip velocity W in the paper as A^(g)
        (Supplementary Material 1 - Empirical parameters - Page 7)

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)
        rho_p (float): density of the particle
        rho_f (float): density of the fluid/gas around the particle
        nu (float): kinematic viscosity of the surrounding fluid/gas
        g (np.ndarray|float): gravitational acceleration

    Returns:
        float: slip velocity
    """
    beta = aspectRatio(a_perp, a_para)
    tau_p = particleResponseTime(a_perp, a_para, rho_p, rho_f, nu)
    A_perp, A_para = translationalResistanceCoefficients(beta)
    A_g = resistanceCoefficient(beta, A_perp, A_para)
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
    a_perp: float,
    a_para: float,
    W: float,
    nu: float,
) -> float:
    """Calculate particle Reynolds number Re_p.

    Formula taken from here: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of leading order reynolds number Re_p (Page 5 - Eq. 11)

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)
        W (float): (instantaneous) slip velocity
         nu (float): kinematic viscosity of the surrounding fluid/gas

    Returns:
        float: zeroth order particle Reynolds number
    """
    a = max(a_perp, a_para)
    Re_p0 = a * W / nu
    return Re_p0

def shapeFactor(beta: float) -> float:
    """Calculate the shape factor/function F_lambda for the interial torque of spheriods.

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
    gamma = np.real(gamma)
    A_perp = 8 * (beta ** 2 - 1) / (3 * beta * ((2 * beta ** 2 - 3) * gamma + 1))
    A_para = 4 * (beta ** 2 - 1) / (3 * beta * ((2 * beta ** 2 - 1) * gamma - 1))
    return A_perp, A_para

def rotationalResistanceCoefficients(a_perp: float, a_para: float) -> Tuple[float, float]:
    """Calculate coefficient of the rotational resistance tensor.

    Equations from: https://doi.org/10.1175/JAS-D-20-0221.1
    Definition of gamma (Supplementary Material 1 - Page 1 - Eq. S1)
    Definition of C_perp & C_para (Supplementary Material 1 - Page 1 - Eq. S4)

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)

    Returns:
        Tuple[float, float]: coefficients in (perpendicular, parallel) direction
    """
    beta = aspectRatio(a_perp=a_perp, a_para=a_para)
    gamma = np.log(beta + np.sqrt(beta ** 2 - 1 + 0j)) / (beta * np.sqrt(beta ** 2 - 1 + 0j))
    gamma = np.real(gamma)
    C_perp = + 8 * a_para * a_perp * (beta ** 4 - 1) \
        / (9 * beta ** 2 * ((2 * beta ** 2 - 1) * gamma - 1))
    C_para = - 8 * a_para * a_perp * (beta ** 2 - 1) \
        / (9 * (gamma - 1) * beta ** 2)
    return C_perp, C_para

def selfConsistencyEqProlateC_F(C_F, A_perp, Re_p0, beta, c0, c1, c2):
    A_CF_Re = 6 * A_perp * C_F * Re_p0 / beta
    sqrtTerm = (- 2 + np.sqrt(4 + A_CF_Re)) / (A_perp * C_F)
    return + 1 - np.sqrt(1 + 0.25 * A_CF_Re) + c1 * sqrtTerm ** 0.687 + c2 * sqrtTerm ** c0

def selfConsistencyEqOblateC_F(C_F, A_para, Re_p0, c0, c1):
    A_CF_Re = A_para * C_F * Re_p0
    sqrt_term = (-2 + np.sqrt(4 + 6 * A_CF_Re)) / (A_para * C_F)
    return (
        A_para * (2 - 1.4142135623730951 * np.sqrt(2 + 3 * A_CF_Re))
        + c0 * sqrt_term ** 0.687 + c1 * sqrt_term ** 0.7512
    )

def selfConsistencyEqProlateC_FDerivative(C_F, A_perp, Re_p0, beta, c0, c1, c2):
    sqrtTerm = (- 2 + np.sqrt(4 + (6 * A_perp * Re_p0 * C_F) / beta))
    term2 = ((3 * Re_p0) / (beta * C_F * np.sqrt(4 + (6 * A_perp * Re_p0 * C_F) / beta)) - sqrtTerm / (A_perp * C_F ** 2))
    return (
        - ((3 * A_perp * 0.25 * Re_p0) / (beta * np.sqrt(1 + 0.25 * (6 * A_perp
        * Re_p0 * C_F) / beta))) + c0 * c2 * (sqrtTerm / (A_perp * C_F)) ** (- 1 + c0)
        * term2 + c1 * 0.687 * (sqrtTerm / (A_perp * C_F)) ** (- 1 + 0.687) * term2
    )

def selfConsistencyEqOblateC_FDerivative(C_F, A_para, Re_p0, c0, c1):
    sqrtTerm = ((-2 + np.sqrt(4 + 6 * A_para * Re_p0 * C_F)) / (A_para * C_F))
    sqrtTerm2 = np.sqrt(2 + 3 * A_para * Re_p0 * C_F)
    return (
        -3 * A_para ** 2 * 1.4142135623730951 * Re_p0 * C_F + (np.sqrt(2)
        - sqrtTerm2) * (c0 * 0.687 * sqrtTerm ** 0.687 + c1 * 0.7512
        * sqrtTerm ** 0.7512)
    ) / (2 * C_F * sqrtTerm2)

def newtonC_F(x0, f, f_prime, args):
    x = x0
    for i in range(5):
        x = x - f(x, *args) / f_prime(x, *args)
    return x

def correctionFactorStokesForce(
    Re_p0: float,
    const: SystemConstants,
    full_solve: bool = False,
) -> float:
    """Calculate correction factor for the Stokes force

    Different papers use 'lambda' or 'beta' for the aspect ratio. Here we use 'beta'.

    Args:
        Re_p0 (float): zeroth-order Reynolds number
        beta (float): ratio of the radii of a spheroid (aspect ratio)
        full_solve (bool, optional): Whether to solve the self-consistency
            equation. Defaults to False.

    Returns:
        float: correction factor for the Stokes force
    """
    # Re_0 upper bound of the actual reynolds number
    # The correction factor C_F for the stokes force for Reynolds numbers > 1
    # full_solve: Whether to solve the self-consistency equation for C_F otherwise an
    #   interpolation formula is used.
    if Re_p0 <= 1:
        C_F = 1
        return C_F

    if const.beta >= 1:  # beta > 1 -> prolate spheroid
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Consider rod-like particles aligned with the steady state direction
        # (phi=pi/2), i.e. consider the drag coefficient (Page 19 - Section 3.3.1 - Eq. 3.4b)
        #   C_{D,90}(Re,beta)=C_{D,Stokes,90}(Re,beta)*f_{d,90}(Re,beta)
        #   C_{D,Stokes,90}(Re,beta) is the analytical drag coefficient (Page 30 - Eq. B5)
        # Take the correction function (Page 30 - Eq. B6b) f_{d,90}(Re,beta) then
        # in our model, we have vgDot=g-A_perp v_g/taup[1+C_F*3/8*ap/beta*A_perp*vg/nu]
        # => f_{d,90}(Re,beta)=1+0.15*ReJFM^0.687+c_{d,5}*log(beta)^c_{d,6}*ReJFM^(c_{d,7}+c_{d,8}*log(beta))
        #                     =1+C_F*3*A_perp*Rep/(8*beta)
        # => C_F=8*beta*(0.15*ReJFM^0.687+c_{d,5}*log(beta)^c_{d,6}*ReJFM^(c_{d,7}+c{d,8}*log(beta)))/(3*A_perp*Rep)
        # TODO: I cannot follow the calculation here.
        if full_solve:
            C_F = newtonC_F(
                x0=1,
                f=selfConsistencyEqProlateC_F,
                f_prime=selfConsistencyEqProlateC_FDerivative,
                args=(
                    const.A_perp, Re_p0, const.beta,
                    const._C_F_prolate_c0,
                    const._C_F_prolate_c1,
                    const._C_F_prolate_c2,
            ))
        else:
            # TODO: Cannot verify formula for RE_JFM
            Re_correction = const.beta  ** (2 / 3) / 2
            Re_JFM = Re_p0 / Re_correction
            # Empirical correlation coefficients (Page 32 - Table 5 c_{d,i})
            # This approximation of C_F is obtained by rearranging the formula given
            # above (cmp. f_{d,90}(Re,beta)=...)
            C_F = 8 * const.beta * (
                + 0.15 * Re_JFM ** 0.687
                + const._c_d[4] * np.log(const.beta) ** const._c_d[5] * Re_JFM ** (const._c_d[6] + const._c_d[7] * np.log(const.beta))
            ) / (3 * const.A_perp * Re_p0)
    else:  # beta < 1 -> oblate spheroid
        # Use Ouchene (2020) for interpolated coefficient : https://doi.org/10.1063/5.0011618
        # Consider disk-like particles aligned with the steady state direction
        # (phi=0), i.e. consider the drag coefficient
        if full_solve:
            C_F = newtonC_F(
                x0=1,
                f=selfConsistencyEqOblateC_F,
                f_prime=selfConsistencyEqOblateC_FDerivative,
                args=(
                    const.A_perp, Re_p0,
                    const._C_F_oblate_c0,
                    const._C_F_oblate_c1,
            ))
        else:
            # TODO: Cannot verify formula for RE_JFM
            Re_correction = max(1, const.beta) / (2 * const.beta ** (1 / 3))
            Re_JFM = Re_p0 / Re_correction
            # K_phi0 is defined in Ouchene (2020) (Page 4 - Eq. 13)
            K_phi0 = (8 / 3) * const.beta ** (-1 / 3) / (
                + 2 * const.beta / (1 - const.beta ** 2)
                + 2 * (1 - 2 * const.beta ** 2) / (1 - const.beta ** 2) ** (3 / 2) * np.arctan(np.sqrt(1 - const.beta ** 2) / const.beta)
            )
            # This approximation of C_F is obtained by rearranging the formula given
            # above (cmp. f_{d,90}(Re,beta)=...) (Fröhlich (2020)) but using the C_{D,0}(Re,beta)
            # from Ouchene (2020) (Page 6 - Eq. 18)
            C_F= 8 * (
                + 0.15 * const.beta ** 95.91 * Re_JFM ** 0.687
                + 0.2927 * (1 - const.beta) ** 0.4374 * Re_JFM ** 0.7512
            ) / K_phi0 / (3 * const.A_para * Re_p0)
    return C_F

def correctionFactorTorque(
    Re_p: float,
    const: SystemConstants,
    do_ouchene_disks=True,
) -> float:
    """Calculate the correction factor for the torque

    Different papers use 'lambda' or 'beta' for the aspect ratio. Here we use 'beta'.

    Args:
        Re_p (float): particle Reynolds number. Can be obtained from
            'correctionFactorStokesForce(full_solve=True)'
        beta (float): ratio of the radii of a spheroid (aspect ratio)
        shape_factor (float): shape factor for the interial torque of a spheriod
        do_ouchene_disks (bool, optional): Whether to use the analytical formula
            for disks from Ouchene (2020). Defaults to True.

    Returns:
        float: correction factor
    """
    if Re_p <= 1:
        C_T = 1
        return C_T

    if const.beta > 1:  # beta > 1 -> prolate spheroid
        C_T = const._C_T_prolate_c1 * Re_p ** (-const._c_t[2]) + const._C_T_prolate_c2 * Re_p ** (-const._C_T_prolate_c0)
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Empirical correlation coefficients (Page 32 - Table 5 c_{t,i})
        # c_t = [0.931, 0.675, 0.162, 0.657, 2.77, 0.178, 0.177]
        # # TODO: Cannot verify this formula
        # Re_correction = beta ** (2 / 3) / 2
        # Re_JFM = Re_p / Re_correction
        # # TODO: Where does this formula come from?
        # # TODO: In Fröhlich (2020) (Page 24 - Eq. 3.16a) where is an additional
        # #       cos(phi) term which would be zero?!
        # C_JFM = (
        #     +c_t[0] * np.log(beta) ** c_t[1] / (Re_JFM ** c_t[2])
        #     +c_t[3] * np.log(beta) ** c_t[4] / (Re_JFM ** (c_t[5] + c_t[6] * np.log(beta)))
        # )
        # # TODO: Where where do these formulas (& corrections) come from?
        # C_correction = np.pi / (2 * beta ** 2)
        # C_Re = C_JFM * C_correction
        # # Original coefficient
        # C_0 = np.abs(shape_factor) / 2
        # C_T = C_Re / C_0
    else:  # beta < 1 -> oblate spheroid
        if do_ouchene_disks:
            C_T =  const._C_T_oblate_c0 * Re_p ** (-0.146)
            # # Use Ouchene (2020) for interpolated coefficient : https://doi.org/10.1063/5.0011618
            # Re_correction = max(1, beta) / (2 * beta ** (1 / 3))
            # Re_PF = Re_p / Re_correction
            # # Ouchene (2020) for phi = pi / 4 (Page 9 - Eq. 24):
            # C_PF = 1.85 * ((1 - beta) / beta) ** 0.832 / (2 * Re_PF ** 0.146)
            # # Coefficient corrected for finite Re
            # C_correction = np.pi * beta / (2 * max(1, beta) ** 3)
            # C_Re = C_correction * C_PF
            # # Original coefficient
            # C_0 = np.abs(shape_factor) / 2
            # C_T = C_Re / C_0
        else:
            # Interpolation with data from Jiang (2021) https://doi.org/10.1103/PhysRevFluids.6.024302
            # TODO: What is 'C_TS'
            # TODO: The data does not seem to be available in the paper
            arr_Re_p = [0, 0.3, 3, 30]
            arr_beta = [1/6, 1/3, 1/2, 1]
            C_TS = np.array([
                [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.0],
                [0.884697478495574, 0.879578485160704, 0.851105581379940, 1.0],
                [0.686980996559064, 0.692434126615873, 0.666082628906040, 1.0],
                [0.491498599164208, 0.505289768071043, 0.488460594531096, 1.0],
            ])
            # performs 2D linear interpolation
            bivariate_obj = RectBivariateSpline(arr_Re_p, arr_beta, C_TS, kx=1, ky=1)
            C_T = np.squeeze(bivariate_obj(Re_p, const.beta))
    return C_T

def explicitCorrectionFactorStokesForce(
    Re_p0: float,
    beta: float,
    full_solve: bool = False,
) -> float:
    """Calculate correction factor for the Stokes force

    Different papers use 'lambda' or 'beta' for the aspect ratio. Here we use 'beta'.

    Args:
        Re_p0 (float): zeroth-order Reynolds number
        beta (float): ratio of the radii of a spheroid (aspect ratio)
        full_solve (bool, optional): Whether to solve the self-consistency
            equation. Defaults to False.

    Returns:
        float: correction factor for the Stokes force
    """
    # Re_0 upper bound of the actual reynolds number
    # The correction factor C_F for the stokes force for Reynolds numbers > 1
    # full_solve: Whether to solve the self-consistency equation for C_F otherwise an
    #   interpolation formula is used.
    if Re_p0 <= 1:
        C_F = 1
        return C_F
    if beta >= 1:  # beta > 1 -> prolate spheroid
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Empirical correlation coefficients (Page 32 - Table 5 c_{d,i})
        c_d = [-0.007, 1.0, 1.17, -0.07, 0.047, 1.14, 0.7, -0.008]
        # Consider rod-like particles aligned with the steady state direction
        # (phi=pi/2), i.e. consider the drag coefficient (Page 19 - Section 3.3.1 - Eq. 3.4b)
        #   C_{D,90}(Re,beta)=C_{D,Stokes,90}(Re,beta)*f_{d,90}(Re,beta)
        #   C_{D,Stokes,90}(Re,beta) is the analytical drag coefficient (Page 30 - Eq. B5)
        # Take the correction function (Page 30 - Eq. B6b) f_{d,90}(Re,beta) then
        # in our model, we have vgDot=g-A_perp v_g/taup[1+C_F*3/8*ap/beta*A_perp*vg/nu]
        # => f_{d,90}(Re,beta)=1+0.15*ReJFM^0.687+c_{d,5}*log(beta)^c_{d,6}*ReJFM^(c_{d,7}+c_{d,8}*log(beta))
        #                     =1+C_F*3*A_perp*Rep/(8*beta)
        # => C_F=8*beta*(0.15*ReJFM^0.687+c_{d,5}*log(beta)^c_{d,6}*ReJFM^(c_{d,7}+c{d,8}*log(beta)))/(3*A_perp*Rep)
        # TODO: I cannot follow the calculation here.
        A_perp, _ = translationalResistanceCoefficients(beta)
        if full_solve:
            def selfConsistencyEqProlateC_F(C_F):
                return + 1 - np.sqrt(1 + 3 * A_perp * C_F * Re_p0 / (2 * beta)) \
                    + 2 * (+ 0.18277810531719724 * beta ** 0.229 * ((- 2 + np.sqrt(4 + 6 * A_perp \
                        * C_F * Re_p0 / beta)) / (A_perp * C_F)) ** 0.687 + 0.75 ** (-c_d[6] - c_d[7] \
                        * np.log(beta)) * c_d[4] * (beta ** (1 / 3) * (- 2 + np.sqrt(4 + 6 * A_perp \
                        * C_F * Re_p0 / beta)) / (A_perp * C_F)) ** (c_d[6] + c_d[7] * np.log(beta)) \
                        * np.log(beta) ** c_d[5] \
                    )

            C_F = optimize.least_squares(fun=selfConsistencyEqProlateC_F, x0=1, bounds=(0.0, np.inf)).x[0]
        else:
            # TODO: Cannot verify formula for RE_JFM
            Re_correction = beta  ** (2 / 3) / 2
            Re_JFM = Re_p0 / Re_correction
            # This approximation of C_F is obtained by rearranging the formula given
            # above (cmp. f_{d,90}(Re,beta)=...)
            C_F = 8 * beta * (
                + 0.15 * Re_JFM ** 0.687
                + c_d[4] * np.log(beta) ** c_d[5] * Re_JFM ** (c_d[6] + c_d[7] * np.log(beta))
            ) / (3 * A_perp * Re_p0)
    else:  # beta < 1 -> oblate spheroid
        # Use Ouchene (2020) for interpolated coefficient : https://doi.org/10.1063/5.0011618
        # Consider disk-like particles aligned with the steady state direction
        # (phi=0), i.e. consider the drag coefficient
        _, A_para = translationalResistanceCoefficients(beta)
        if full_solve:
            # TODO: Where does this formula come from?
            def selfConsistencyEqOblateC_F(C_F):
                return (
                    A_para * (2 - np.sqrt(2) * np.sqrt(2 + 3 * A_para * C_F * Re_p0)) \
                    + 0.7311124212687891 * beta ** 96.47233333333332  * ((-2 + np.sqrt(4 + 6 * A_para \
                    * C_F * Re_p0)) / (A_para * C_F)) ** 0.687 + 1.453237823359436 \
                    * (1 - beta) ** 0.4374 * beta ** 0.5837333333333332 * ((-2 + np.sqrt(4 + 6 * A_para \
                    * C_F * Re_p0)) / (A_para * C_F)) ** 0.7512
                )

            C_F=optimize.least_squares(fun=selfConsistencyEqOblateC_F, x0=1, bounds=(0.0, np.inf)).x[0]
        else:
            # TODO: Cannot verify formula for RE_JFM
            Re_correction = max(1, beta) / (2 * beta ** (1 / 3))
            Re_JFM = Re_p0 / Re_correction
            # K_phi0 is defined in Ouchene (2020) (Page 4 - Eq. 13)
            # TODO: The formula in the paper does not agree with the formula here
            # (copied from the MATLAB-script)
            K_phi0 = A_para / beta ** (1 / 3)
            # This approximation of C_F is obtained by rearranging the formula given
            # above (cmp. f_{d,90}(Re,beta)=...) (Fröhlich (2020)) but using the C_{D,0}(Re,beta)
            # from Ouchene (2020) (Page 6 - Eq. 18)
            C_F= 8 * (
                + 0.15 * beta ** 95.91 * Re_JFM ** 0.687
                + 0.2927 * (1 - beta) ** 0.4374 * Re_JFM ** 0.7512
            ) / K_phi0 / (3 * A_para * Re_p0)
    return C_F

def explicitCorrectionFactorTorque(
    Re_p: float,
    beta: float,
    shape_factor: float,
    do_ouchene_disks=True,
) -> float:
    """Calculate the correction factor for the torque

    Different papers use 'lambda' or 'beta' for the aspect ratio. Here we use 'beta'.

    Args:
        Re_p (float): particle Reynolds number. Can be obtained from
            'correctionFactorStokesForce(full_solve=True)'
        beta (float): ratio of the radii of a spheroid (aspect ratio)
        shape_factor (float): shape factor for the interial torque of a spheriod
        do_ouchene_disks (bool, optional): Whether to use the analytical formula
            for disks from Ouchene (2020). Defaults to True.

    Returns:
        float: correction factor
    """
    if Re_p <= 1:
        C_T = 1
        return C_T

    if beta > 1:  # beta > 1 -> prolate spheroid
        # The following formulas are discussed in Fröhlich JFM 901 (2020):
        #   https://doi.org/10.1017/jfm.2020.482
        # Empirical correlation coefficients (Page 32 - Table 5 c_{t,i})
        c_t = [0.931, 0.675, 0.162, 0.657, 2.77, 0.178, 0.177]

        # TODO: Cannot verify this formula
        Re_correction = beta ** (2 / 3) / 2
        Re_JFM = Re_p / Re_correction
        # TODO: Where does this formula come from?
        # TODO: In Fröhlich (2020) (Page 24 - Eq. 3.16a) where is an additional
        #       cos(phi) term which would be zero?!
        C_JFM = (
            +c_t[0] * np.log(beta) ** c_t[1] / (Re_JFM ** c_t[2])
            +c_t[3] * np.log(beta) ** c_t[4] / (Re_JFM ** (c_t[5] + c_t[6] * np.log(beta)))
        )
        # TODO: Where where do these formulas (& corrections) come from?
        C_correction = np.pi / (2 * beta ** 2)
        C_Re = C_JFM * C_correction
        # Original coefficient
        C_0 = np.abs(shape_factor) / 2
        C_T = C_Re / C_0
    else:  # beta < 1 -> oblate spheroid
        if do_ouchene_disks:
            # Use Ouchene (2020) for interpolated coefficient : https://doi.org/10.1063/5.0011618
            Re_correction = max(1, beta) / (2 * beta ** (1 / 3))
            Re_PF = Re_p / Re_correction
            # Ouchene (2020) for phi = pi / 4 (Page 9 - Eq. 24):
            C_PF = 1.85 * ((1 - beta) / beta) ** 0.832 / (2 * Re_PF ** 0.146)
            # Coefficient corrected for finite Re
            C_correction = np.pi * beta / (2 * max(1, beta) ** 3)
            C_Re = C_correction * C_PF
            # Original coefficient
            C_0 = np.abs(shape_factor) / 2
            C_T = C_Re / C_0
        else:
            # Interpolation with data from Jiang (2021) https://doi.org/10.1103/PhysRevFluids.6.024302
            # TODO: What is 'C_TS'
            # TODO: The data does not seem to be available in the paper
            arr_Re_p = [0, 0.3, 3, 30]
            arr_beta = [1/6, 1/3, 1/2, 1]
            C_TS = np.array([
                [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.0],
                [0.884697478495574, 0.879578485160704, 0.851105581379940, 1.0],
                [0.686980996559064, 0.692434126615873, 0.666082628906040, 1.0],
                [0.491498599164208, 0.505289768071043, 0.488460594531096, 1.0],
            ])
            # performs 2D linear interpolation
            bivariate_obj = RectBivariateSpline(arr_Re_p, arr_beta, C_TS, kx=1, ky=1)
            C_T = np.squeeze(bivariate_obj(Re_p, beta))
    return C_T

def particleInteriaTensorCoefficents(
    a_perp: float,
    a_para: float,
    particle_mass: float,
) -> Tuple[float, float]:
    """Calculate unique matrix elements in of the particle interia tensor
    (in particle coordinate frame)

    Equations from: https://doi.org/10.1175/JAS-D-20-0221.1
    Definitions of I_para & I_perp (Supplementary Material 1 - Page 1 - Eq. S5)

    Args:
        a_perp (float): particle radius (perpendicular to symmetry axis)
        a_para (float): particle radius (parallel      to symmetry axis)
        particle_mass (float): mass of the particle

    Returns:
        Tuple[float, float]: coefficients in the particle interia tensor
    """
    beta = aspectRatio(a_perp=a_perp, a_para=a_para)
    J_perp = (1 / 5) * particle_mass * a_perp ** 2 * (1 + beta ** 2)
    J_para = (2 / 5) * particle_mass * a_perp ** 2
    return J_perp, J_para

def systemDynamics(
    t: float,
    state: np.ndarray,
    const: SystemConstants,
) -> np.ndarray:
    """Compute the system dynamics (differential equation) for a spheriod in a
    quinesent flow.

    Args:
        t (float): time t in thesimulation
        state (np.ndarray): all state variables [x, v, n, omega] in order in a
            single vector.
        const (SystemConstants): collection of constants required for the computation.

    Returns:
        np.ndarray: derivative of the given state vector
    """
    # extract state variables
    x     = state[0:3]  # particle position
    v     = state[3:6]  # particle velocity
    n     = state[6:9]  # unit vector parallel to the symmetry axis of the particle
    omega = state[9:12] # angular velocity of the particle
    n = n / np.linalg.norm(n)

    v_mag = np.linalg.norm(v)
    v_hat = v / v_mag
    # A = Translation resistance tensor
    # C = Rotation resistance tensor
    A = const.A_perp * np.eye(3) + const._A_diff * np.outer(n, n)
    C = const.C_perp * np.eye(3) + const._C_diff * np.outer(n, n)
    # Stokes force + correction for higher particle Reynolds numbers (Re_p ~ 1 - 30)
    F_h0 = const._fac_TF_h0 * (A @ v)
    F_h1 = const._fac_F_h1 * v_mag * (3 * A - np.eye(3) * (v_hat @ A @ v_hat)) @ A @ v
    # Torque + correction for higher particle Reynolds numbers (Re_p ~ 1 - 30)
    T_h0 = const._fac_TF_h0 * C @ omega
    T_h1 = const._fac_T_h1 * (n @ v) * np.cross(n, v)
    # Compute correction factors
    Re_p0 = const._fac_Re_p0 * v_mag
    C_F = correctionFactorStokesForce(Re_p0, const, full_solve=True)
    v_g_star = const._fac1_v_g_star * (np.sqrt(1 + const._fac2_v_g_star * C_F) - 1) / C_F
    Re_p = const._fac_Re_p0 * v_g_star
    C_T = correctionFactorTorque(Re_p, const)
    # Full terms for torque and stokes force
    F_h = F_h0 + C_F * F_h1
    T_h = T_h0 + C_T * T_h1
    # Particle interia tensor
    J_inverse = (
        const._J_diff * np.outer(n, n) + const.J_para * np.eye(3)
    ) / (const.J_perp * const.J_para)
    dJ_pdt = - const._J_diff * (
        np.outer(n, np.cross(omega, n)) + np.outer(np.cross(omega, n), n)
    )
    # Derivatives of the system variables
    dxdt = v
    dvdt = F_h / const.m_p + const.g
    dndt = np.cross(omega, n)
    domegadt = J_inverse @ (T_h - dJ_pdt @ omega)
    state_derivative = np.concat([dxdt, dvdt, dndt, domegadt])
    return state_derivative
