from pathlib import Path
import ctypes
from ctypes import c_double, c_size_t, c_int
from ctypes import pointer, byref, POINTER, cast
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import dynamics

LIBARY_PATH = Path("cpp-impl/solve.so")
CPP_STRUCT_MEMBERS = [
    "beta", "curly_A_F", "curly_A_T",
    "A_para", "A_perp", "C_para", "C_perp", "J_para", "J_perp",
    "Re_p0", "_fac_Re_p", "_fac1_v_g_star", "_fac2_v_g_star",
    "_C_T_prolate_c0", "_C_T_prolate_c1", "_C_T_prolate_c2",
    "_C_T_oblate_c0",
]
C_DOUBLE_PTR = POINTER(c_double)

class CppConstantsStruct(ctypes.Structure):
    _fields_ = [(name, c_double) for name in CPP_STRUCT_MEMBERS]

class CppConfig:
    def __init__(self, const: dynamics.SystemConstants):
        self.cpp_constants_struct = CppConfig._initConstants(const)

        self.c_dll = np.ctypeslib.load_library(LIBARY_PATH, ".")
        self.c_compute_C_F = self.c_dll.correctionFactorStokesForce
        self.c_compute_C_T = self.c_dll.correctionFactorTorque
        self.c_solve_dynamics = self.c_dll.solveDynamics
        self.c_compute_C_F.restype = c_double
        self.c_compute_C_T.restype = c_double
        self.c_solve_dynamics.restype = None
        self.c_compute_C_F.argtypes = [
            c_double,
            POINTER(CppConstantsStruct),
        ]
        self.c_compute_C_T.argtypes = [
            c_double,
            POINTER(CppConstantsStruct),
        ]
        self.c_solve_dynamics.argtypes = [
            POINTER(C_DOUBLE_PTR),
            POINTER(c_size_t),
            C_DOUBLE_PTR,
            C_DOUBLE_PTR,
            c_size_t,
            C_DOUBLE_PTR,
            POINTER(CppConstantsStruct),
            c_double,
            c_double,
            c_int,
        ]


    def _initConstants(const: dynamics.SystemConstants) -> CppConstantsStruct:
        struct = CppConstantsStruct()
        for name in CPP_STRUCT_MEMBERS:
            value = getattr(const, name, None)
            if value is not None:
                setattr(struct, name, c_double(value))
            else:
                raise TypeError(f"Missing {name}")
        return struct

    def correctionFactorStokesForce(self, Re_p0: float) -> float:
        return self.c_compute_C_F(
            Re_p0, byref(self.cpp_constants_struct)
        )

    def correctionFactorTorque(self, Re_p0: float) -> float:
        return self.c_compute_C_T(
            Re_p0, byref(self.cpp_constants_struct)
        )

def _numpyArray2DToCtypesPtr(arr):
    ct_arr = np.ctypeslib.as_ctypes(arr)
    ptr_arr = C_DOUBLE_PTR * arr.shape[0]
    ptr_ptr_arr = cast(
        ptr_arr(*(cast(row, C_DOUBLE_PTR) for row in ct_arr)),
        POINTER(C_DOUBLE_PTR))
    return ptr_ptr_arr

def solveDynamics(
        y0: NDArray[np.float64],
        const: dynamics.SystemConstants,
        t_eval: NDArray[np.float64] = None,
        t_span: Tuple[float, float] = None,
        rel_tol: float=1e-6, abs_tol: float=1e-6, event_type: int=0
    ):
    if t_eval is None and t_span is None:
        raise ValueError("Must specify t_eval or t_span")

    config = CppConfig(const)
    result_size = 1 if t_eval is None else t_eval.size
    result = np.empty(shape=(result_size, 12), dtype=np.float64)

    if t_eval is not None:
        t_eval_ptr = t_eval.ctypes.data_as(C_DOUBLE_PTR)
        t_span_ptr = None
    else:
        t_span_arr = np.array(t_span, dtype=np.float64)
        t_span_ptr = t_span_arr.ctypes.data_as(C_DOUBLE_PTR)
        t_eval_ptr = None
    y_eval = _numpyArray2DToCtypesPtr(result)
    num_evaluations = c_size_t(0)
    config.c_solve_dynamics(
        y_eval,
        byref(num_evaluations),
        t_span_ptr,
        t_eval_ptr,
        c_size_t(result_size),
        y0.ctypes.data_as(C_DOUBLE_PTR),
        byref(config.cpp_constants_struct),
        c_double(rel_tol),
        c_double(abs_tol),
        c_int(event_type)
    )
    if t_eval is None:
        return t_span_arr[1], result[0]
    return t_eval[:num_evaluations.value], result[:num_evaluations.value]
