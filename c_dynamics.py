from pathlib import Path
import ctypes
from ctypes import c_double, c_size_t, pointer, byref, POINTER, cast
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import dynamics

LIBARY_PATH = Path("cpp-impl/solve.so")
CPP_STRUCT_MEMBERS = [
    "m_p", "beta", "A_perp", "C_perp", "F_lambda", "J_perp",
    "J_para", "_A_diff", "_C_diff", "_fac_TF_h0", "_fac_F_h1",
    "_fac_T_h1", "_fac_Re_p0", "_fac1_v_g_star", "_fac2_v_g_star",
    "_J_diff", "_C_F_prolate_c0", "_C_F_prolate_c1", "_C_F_prolate_c2",
    "_C_F_oblate_c0", "_C_F_oblate_c1", "_C_T_prolate_c0", "_C_T_prolate_c1",
    "_C_T_prolate_c2", "_C_T_oblate_c0", "g_x", "g_y", "g_z",
]
C_DOUBLE_PTR = POINTER(c_double)

class CppConstantsStruct(ctypes.Structure):
    _fields_ = [(name, c_double) for name in CPP_STRUCT_MEMBERS]

class CppConfig:
    def __init__(self, const: dynamics.SystemConstants):
        self.cpp_constants_struct = CppConfig.initConstants(const)

        c_dll = np.ctypeslib.load_library(LIBARY_PATH, ".")
        self.func_solve = c_dll.solveDynamics
        self.func_solve.restype = None
        self.func_solve.argtypes = [
            POINTER(C_DOUBLE_PTR),
            POINTER(c_size_t),
            C_DOUBLE_PTR,
            c_size_t,
            C_DOUBLE_PTR,
            POINTER(CppConstantsStruct),
            c_double,
            c_double
        ]

    def initConstants(const: dynamics.SystemConstants) -> CppConstantsStruct:
        struct = CppConstantsStruct()
        for name in CPP_STRUCT_MEMBERS:
            value = getattr(const, name, None)
            if value is not None:
                setattr(struct, name, c_double(value))
        for i, axis in enumerate(["x", "y", "z"]):
            setattr(struct, f"g_{axis}", c_double(const.g[i]))
        struct.g_pad = c_double(0.0)
        return struct

def solveDynamics(
        y0: NDArray[np.float64],
        t_eval: NDArray[np.float64],
        const: dynamics.SystemConstants,
        rel_tol=1e-6, abs_tol=1e-6
    ):
    config = CppConfig(const)
    result = np.empty(shape=(t_eval.size, 12), dtype=np.float64)

    ct_arr = np.ctypeslib.as_ctypes(result)
    ptr_arr = C_DOUBLE_PTR * t_eval.size
    y_eval = cast(
        ptr_arr(*(cast(row, C_DOUBLE_PTR) for row in ct_arr)),
        POINTER(C_DOUBLE_PTR))
    num_evaluations = c_size_t(0)
    len_t_eval = c_size_t(t_eval.size)
    rel_tol = c_double(rel_tol)
    abs_tol = c_double(abs_tol)
    config.func_solve(
        y_eval,
        byref(num_evaluations),
        t_eval.ctypes.data_as(C_DOUBLE_PTR),
        len_t_eval,
        y0.ctypes.data_as(C_DOUBLE_PTR),
        byref(config.cpp_constants_struct),
        rel_tol,
        abs_tol
    )
    return result[:num_evaluations.value]
