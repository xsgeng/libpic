from typing import List
import numpy as np

def current_deposition_cpu(
    rho_list: List[np.ndarray],
    jx_list: List[np.ndarray],
    jy_list: List[np.ndarray],
    jz_list: List[np.ndarray],
    x0_list: List[float],
    y0_list: List[float],
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    ux_list: List[np.ndarray],
    uy_list: List[np.ndarray],
    uz_list: List[np.ndarray],
    inv_gamma_list: List[np.ndarray],
    is_dead_list: List[np.ndarray],
    w_list: List[np.ndarray],
    npatches: int,
    dx: float,
    dy: float,
    dt: float,
    q: float
) -> None: ...
