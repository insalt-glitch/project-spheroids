from pathlib import Path
import ctypes
import numpy as np
from numpy.typing import NDArray
import dynamics

LIBARY_PATH = Path("cpp-impl/dynamics.so")
CPP_STRUCT_MEMBERS = [
    "m_p", "beta", "A_perp", "C_perp", "F_lambda", "J_perp",
    "J_para", "_A_diff", "_C_diff", "_fac_TF_h0", "_fac_F_h1",
    "_fac_T_h1", "_fac_Re_p0", "_fac1_v_g_star", "_fac2_v_g_star",
    "_J_diff", "_C_F_prolate_c0", "_C_F_prolate_c1", "_C_F_prolate_c2",
    "_C_F_oblate_c0", "_C_F_oblate_c1", "_C_T_prolate_c0", "_C_T_prolate_c1",
    "_C_T_prolate_c2", "_C_T_oblate_c0", "g_x", "g_y", "g_z",
]

class CppConstantsStruct(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in CPP_STRUCT_MEMBERS]

class CppConfig:
    def __init__(self, const: dynamics.SystemConstants):
        self.cpp_constants_struct = CppConfig.initConstants(const)

        c_dll = np.ctypeslib.load_library(LIBARY_PATH, ".")
        self.func_dynamics = c_dll.systemDynamics
        self.func_dynamics.restype = None
        self.func_dynamics.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(CppConstantsStruct),
        ]

    def initConstants(const: dynamics.SystemConstants) -> CppConstantsStruct:
        struct = CppConstantsStruct()
        for name in CPP_STRUCT_MEMBERS:
            value = getattr(const, name, None)
            if value is not None:
                setattr(struct, name, ctypes.c_double(value))
        for i, axis in enumerate(["x", "y", "z"]):
            setattr(struct, f"g_{axis}", ctypes.c_double(const.g[i]))
        struct.g_pad = ctypes.c_double(0.0)
        return struct

def systemDynamics(
    t: float,
    state: NDArray[np.float64],
    config: CppConfig,
) -> NDArray[np.float64]:
    derivatives = np.empty(shape=(12,), dtype=np.float64)
    config.func_dynamics(
        derivatives.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.pointer(config.cpp_constants_struct),
    )
    return derivatives
