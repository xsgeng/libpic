""" Boundaries for the FDTD Grid.

Available Boundaries:

 - PeriodicBoundary
 - PML

"""
## Imports
import numpy as np
from numba import njit, prange
from scipy.constants import c, epsilon_0

## Boundary Conditions [base class]
class Boundary:
    ...

## Perfectly Matched Layer (PML)
class PML(Boundary):
    """ A perfectly matched layer (PML)

    PML is an extended area between real simulation grids and ghost/guard cells

    """

    def __init__(
        self, 
        dimensions : list[int],
        dx : float,
        dy : float,
        n_guard : int,
        thickness : int=6, 
        kappa_max : float=20.0, 
        a_max:float=0.15, 
        sigma_max:float=0.7
    ):
        """ Perfectly Matched Layer

        Args:
            dimensions: list of inteters
                the dimensions of the main grid
            thickness: 
                the thickness of the PML
            kappa_max: 
                the maximum value of the electric conductivity
            a_max: 
                the maximum value of the parameter a
            sigma_max: 
                the maximum value of the parameter sigma
        """

        self.dimension = len(dimensions)
        assert self.dimension <=3
        self.dimensions = dimensions
        self.dx = dx
        self.dy = dy
        self.n_guard = n_guard

        self.thickness = thickness
        self.kappa_max = kappa_max
        self.a_max = a_max
        self.sigma_max = sigma_max

        self.cpml_m = 3
        self.cpml_ma = 1
        self.sigma_maxval = sigma_max * c * 0.8* (self.cpml_m + 1.0) / dx


        self.kappa_ex = np.ones(self.dimensions[0])
        self.kappa_bx = np.ones(self.dimensions[0])
        self.a_ex = np.zeros(self.dimensions[0])
        self.a_bx = np.zeros(self.dimensions[0])
        self.sigma_ex = np.zeros(self.dimensions[0])
        self.sigma_bx = np.zeros(self.dimensions[0])


        self.kappa_ey = np.ones(self.dimensions[1])
        self.kappa_by = np.ones(self.dimensions[1])
        self.a_ey = np.zeros(self.dimensions[1])
        self.a_by = np.zeros(self.dimensions[1])
        self.sigma_ey = np.zeros(self.dimensions[1])
        self.sigma_by = np.zeros(self.dimensions[1])

        self.init_parameters()



    def init_parameters(self):
        raise NotImplementedError

    def advance_e_currents(self):
        """ Advance the electric currents in the PML

        This method advances the electric currents in the PML by one time step.
        """
        raise NotImplementedError
    
    
@njit(parallel=True)
def update_efield_cpml_2d(
    ex, ey, ez, 
    bx, by, bz, 
    jx, jy, jz, 
    kappa_ex, kappa_ey,
    dx, dy, dt, 
    nx, ny, n_guard
):
    bfactor = dt * c**2
    jfactor = dt / epsilon_0
    for j in prange(-1, ny):
        bfactor_y = bfactor / kappa_ey[j]
        for i in range(-1, nx):
            bfactor_x = bfactor / kappa_ex[i]
            ex[i, j] += bfactor_y * ( (bz[i, j] - bz[i, j-1]) / dy) - jfactor * jx[i, j]
            ey[i, j] += bfactor_x * (-(bz[i, j] - bz[i-1, j]) / dx) - jfactor * jy[i, j]
            ez[i, j] += bfactor_x * ( (by[i, j] - by[i-1, j]) / dx) \
                      - bfactor_y * ( (bx[i, j] - bx[i, j-1]) / dy) - jfactor * jz[i, j]


@njit(parallel=True)
def update_bfield_cpml_2d(
    ex, ey, ez, 
    bx, by, bz, 
    kappa_bx, kappa_by,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for j in prange(-1, ny):
        efactor_y = dt / kappa_by[j]
        for i in range(-1, nx):
            efactor_x = dt / kappa_bx[i]
            bx[i, j] -= efactor_y * ( (ez[i, j+1] - ez[i, j]) / dy)
            by[i, j] -= efactor_x * (-(ez[i+1, j] - ez[i, j]) / dx)
            bz[i, j] -= efactor_x * ( (ey[i+1, j] - ey[i, j]) / dx) \
                      - efactor_y * ( (ex[i, j+1] - ex[i, j]) / dy)


def _init_parameters(pos_m, pos_ma, kappa_max, sigma_maxval, a_max, s: slice, kappa, sigma, a):
    kappa[s] = 1 + (kappa_max - 1) * pos_m
    sigma[s] = sigma_maxval * pos_m
    a[s] = a_max * pos_ma

@njit
def update_psi_x_and_e(kappa, sigma, a, ny, dt, dx, start, stop, by, bz, ey, ez, psi_ey_x, psi_ez_x):
    fac = dt * c**2
    for iy in prange(ny):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx

            psi_ey_x[ipos, iy] = bcoeff * psi_ey_x[ipos, iy] \
                + ccoeff_d * (bz[ipos, iy] - bz[ipos-1, iy])
            psi_ez_x[ipos, iy] = bcoeff * psi_ez_x[ipos, iy] \
                + ccoeff_d * (by[ipos, iy] - by[ipos-1, iy])

            ey[ipos, iy] -= fac * psi_ey_x[ipos, iy]
            ez[ipos, iy] += fac * psi_ez_x[ipos, iy]

@njit
def update_psi_x_and_b(kappa, sigma, a, ny, dt, dx, start, stop, ey, ez, by, bz, psi_by_x, psi_bz_x):
    fac = dt
    for iy in prange(ny):
        for ipos in range(start, stop):
            kappa_ = kappa[ipos]
            sigma_ = sigma[ipos]
            acoeff = a[ipos]
            bcoeff = np.exp(-(sigma_/kappa_ + acoeff) * dt)
            ccoeff_d = (bcoeff - 1) * sigma_ / kappa_ / (sigma_ + kappa_*acoeff) / dx

            psi_by_x[ipos, iy] = bcoeff * psi_by_x[ipos, iy] \
                + ccoeff_d * (ez[ipos+1, iy] - ez[ipos, iy])
            psi_bz_x[ipos, iy] = bcoeff * psi_bz_x[ipos, iy] \
                + ccoeff_d * (ey[ipos+1, iy] - ey[ipos, iy])

            by[ipos, iy] += fac * psi_by_x[ipos, iy]
            bz[ipos, iy] -= fac * psi_bz_x[ipos, iy]


class PMLXmin(PML):
    def init_parameters(self):
        # runs from 1.0 to nearly 0.0 (actually 0.0 at cpml_thickness+1)
        pos = 1.0 - np.arange(self.thickness, dtype=float) / self.thickness
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma
        cpml_slice = np.s_[:self.thickness]
        _init_parameters(pos_m, pos_ma, self.kappa_max, self.sigma_maxval, self.a_max,
                         cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 1.0 to nearly 0.0 on the half intervals
        # 1.0 at ix_glob=1-1/2 and 0.0 at ix_glob=cpml_thickness+1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5) / self.thickness
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma
        _init_parameters(pos_m, pos_ma, self.kappa_max, self.sigma_maxval, self.a_max,
                         cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)
        
        self.psi_ey_x = np.zeros(self.dimensions)
        self.psi_ez_x = np.zeros(self.dimensions)
        self.psi_by_x = np.zeros(self.dimensions)
        self.psi_bz_x = np.zeros(self.dimensions)

    def advance_e_currents(self, ey, ez, by, bz, dt):
        update_psi_x_and_e(self.kappa_ex, self.sigma_ex, self.a_ex, self.dimensions[1], dt, self.dx, 0, self.thickness, 
                           by, bz, ey, ez, self.psi_ey_x, self.psi_ez_x)

    def advance_b_currents(self, ey, ez, by, bz, dt):
        update_psi_x_and_b(self.kappa_bx, self.sigma_bx, self.a_bx, self.dimensions[1], dt, self.dx, 0, self.thickness, 
                           ey, ez, by, bz, self.psi_by_x, self.psi_bz_x)


class PMLXmax(PML):
    def init_parameters(self):
        # runs from nearly 0.0 (actually 0.0 at cpml_thickness+1) to 1.0
        pos = 1.0 - np.arange(self.thickness, dtype=float)[::-1] / self.thickness
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma

        cpml_slice = np.s_[-2*self.n_guard-self.thickness:-2*self.n_guard]
        _init_parameters(pos_m, pos_ma, self.kappa_max, self.sigma_maxval, self.a_max,
                         cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 0.0 to nearly 1.0 on the half intervals
        # 0.0 at ix_glob=cpml_thickness+1/2 and 1.0 at ix_glob=1-1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5)[::-1] / self.thickness
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma
        cpml_slice = np.s_[-2*self.n_guard-self.thickness-1:-2*self.n_guard-1]
        _init_parameters(pos_m, pos_ma, self.kappa_max, self.sigma_maxval, self.a_max,
                         cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)

    def advance_e_currents(self):
        ...
    def advance_b_currents(self):
        ...


def test_PMLXmin():
    nx = 300
    ny = 300
    bound = PMLXmin(dimensions=(nx, ny), dx=0.1, dy=0.1, n_guard=5)
    ey = np.zeros((nx, ny))
    ez = np.zeros((nx, ny))
    by = np.zeros((nx, ny))
    bz = np.zeros((nx, ny))
    bound.advance_e_currents(ey, ez, by, bz, 0.1/c)


if __name__ == "__main__":
    test_PMLXmin()