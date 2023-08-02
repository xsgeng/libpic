from scipy.constants import m_e, c, pi, epsilon_0, hbar, e
from math import sqrt
from numba import njit, prange

def boris_inline( ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q, m, dt ) :

    efactor = q*dt/(2*m*c)
    bfactor = q*dt/(2*m)

    # E field
    ux_minus = ux + efactor * Ex
    uy_minus = uy + efactor * Ey
    uz_minus = uz + efactor * Ez
    # B field
    inv_gamma_minus = 1 / sqrt(1 + ux_minus**2 + uy_minus**2 + uz_minus**2)
    Tx = bfactor * Bx * inv_gamma_minus
    Ty = bfactor * By * inv_gamma_minus
    Tz = bfactor * Bz * inv_gamma_minus
    
    ux_prime = ux_minus + uy_minus * Tz - uz_minus * Ty
    uy_prime = uy_minus + uz_minus * Tx - ux_minus * Tz
    uz_prime = uz_minus + ux_minus * Ty - uy_minus * Tx

    Tfactor = 2 / (1 + Tx**2 + Ty**2 + Tz**2)
    Sx = Tfactor * Tx
    Sy = Tfactor * Ty
    Sz = Tfactor * Tz

    ux_plus = ux_minus + uy_prime * Sz - uz_prime * Sy
    uy_plus = uy_minus + uz_prime * Sx - ux_prime * Sz
    uz_plus = uz_minus + ux_prime * Sy - uy_prime * Sx

    ux_new = ux_plus + efactor * Ex
    uy_new = uy_plus + efactor * Ey
    uz_new = uz_plus + efactor * Ez
    inv_gamma_new = 1 / sqrt(1 + ux_new**2 + uy_new**2 + uz_new**2)
    return ux_new, uy_new, uz_new, inv_gamma_new

@njit(cache=True)
def push_position_2d( x, y, ux, uy, inv_gamma, N, pruned, dt):
    """
    Advance the particles' positions over `dt` using the momenta `ux`, `uy`, `uz`,
    """
    # Timestep, multiplied by c
    cdt = c*dt

    for ip in prange(N) :
        if pruned[ip]:
            continue
        x[ip] += cdt * inv_gamma[ip] * ux[ip]
        y[ip] += cdt * inv_gamma[ip] * uy[ip]


boris_cpu = njit(boris_inline, inline="always")

subsize = 32
@njit(cache=True)
def boris( ux, uy, uz, inv_gamma, ex_part, ey_part, ez_part, bx_part, by_part, bz_part, q, m, npart, pruned, dt ) :
    for ip in range(npart):
        if pruned[ip]:
            continue

        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = boris_cpu(
            ux[ip], uy[ip], uz[ip], ex_part[ip], ey_part[ip], ez_part[ip], bx_part[ip], by_part[ip], bz_part[ip], q, m, dt)