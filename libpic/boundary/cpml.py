import numpy as np
from numba import njit, prange
from scipy.constants import c, epsilon_0

from libpic.fields import Fields, Fields2D, Fields3D


class Boundary:
    ...

class PML(Boundary):
    """ A perfectly matched layer (PML)

    PML is an extended area between real simulation grids and ghost/guard cells

    """

    def __init__(
        self, 
        fields: Fields,
        thickness : int=6, 
        kappa_max : float=20.0, 
        a_max:float=0.15, 
        sigma_max:float=0.7
    ) -> None:
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

        self.fields = fields
        self.nx = fields.nx
        self.dx = fields.dx
        self.n_guard = fields.n_guard

        self.thickness = thickness
        self.kappa_max = kappa_max
        self.a_max = a_max
        self.sigma_max = sigma_max

        self.cpml_m: int = 3
        self.cpml_ma: int = 1
        self.sigma_maxval: float = sigma_max * c * 0.8* (self.cpml_m + 1.0) / self.dx


        if isinstance(fields, Fields2D):
            self.ny = fields.ny
            self.dy = fields.dy
            self.dimensions = (fields.nx, fields.ny)
            shapex = fields.nx
            shapey = fields.ny
            shapez = 0
        # if isinstance(fields, Fields3D):
        #     self.ny = fields.ny
        #     self.nz = fields.nz
        #     self.dy = fields.dy
        #     self.dz = fields.dz
        #     self.dimensions = (fields.nx, fields.ny, fields.nz)
        #     shapex = (fields.nx, fields.ny)
        #     shapey = (fields.ny, fields.nz)
        #     shapez = (fields.nz, fields.nx)

        self.kappa_ex = np.ones(shapex)
        self.kappa_bx = np.ones(shapex)
        self.a_ex = np.zeros(shapex)
        self.a_bx = np.zeros(shapex)
        self.sigma_ex = np.zeros(shapex)
        self.sigma_bx = np.zeros(shapex)


        self.kappa_ey = np.ones(shapey)
        self.kappa_by = np.ones(shapey)
        self.a_ey = np.zeros(shapey)
        self.a_by = np.zeros(shapey)
        self.sigma_ey = np.zeros(shapey)
        self.sigma_by = np.zeros(shapey)

        self.kappa_ez = np.ones(shapez)
        self.kappa_bz = np.ones(shapez)
        self.a_ez = np.zeros(shapez)
        self.a_bz = np.zeros(shapez)
        self.sigma_ez = np.zeros(shapez)
        self.sigma_bz = np.zeros(shapez)

        self.init_parameters()


    def init_parameters(self) -> None:
        """
        Init parameters, will be called by inherted PMLs.
        """
        raise NotImplementedError

    def init_coefficents(self, pos: np.ndarray, s: slice, kappa: np.ndarray, sigma: np.ndarray, a: np.ndarray) -> None:
        pos_m = pos**self.cpml_m
        pos_ma = (1 - pos)**self.cpml_ma

        kappa[s] = 1 + (self.kappa_max - 1) * pos_m
        sigma[s] = self.sigma_maxval * pos_m
        a[s] = self.a_max * pos_ma

    def advance_e_currents(self, dt):
        """ Advance the CPML psi_e """
        raise NotImplementedError
    
    def advance_b_currents(self, dt):
        """ Advance the CPML psi_b """
        raise NotImplementedError

class PMLX(PML):
    def __init__(self, fields: Fields, thickness: int = 6, kappa_max: float = 20, a_max: float = 0.15, sigma_max: float = 0.7) -> None:
        super().__init__(fields, thickness, kappa_max, a_max, sigma_max)
        self.psi_ey_x = np.zeros(self.dimensions)
        self.psi_ez_x = np.zeros(self.dimensions)
        self.psi_by_x = np.zeros(self.dimensions)
        self.psi_bz_x = np.zeros(self.dimensions)

    def advance_e_currents(self, dt):
        update_psi_x_and_e(self.kappa_ex, self.sigma_ex, self.a_ex, self.ny, dt, self.dx, self.efield_start, self.efield_end, 
                           self.fields.by, self.fields.bz, self.fields.ey, self.fields.ez, self.psi_ey_x, self.psi_ez_x)

    def advance_b_currents(self, dt):
        update_psi_x_and_b(self.kappa_bx, self.sigma_bx, self.a_bx, self.ny, dt, self.dx, self.bfield_start, self.bfield_end, 
                           self.fields.ey, self.fields.ez, self.fields.by, self.fields.bz, self.psi_by_x, self.psi_bz_x)

class PMLY(PML):
    def __init__(self, fields: Fields, thickness: int = 6, kappa_max: float = 20, a_max: float = 0.15, sigma_max: float = 0.7) -> None:
        super().__init__(fields, thickness, kappa_max, a_max, sigma_max)
        self.psi_ex_y = np.zeros(self.dimensions)
        self.psi_ez_y = np.zeros(self.dimensions)
        self.psi_bx_y = np.zeros(self.dimensions)
        self.psi_bz_y = np.zeros(self.dimensions)

    # def advance_e_currents(self, dt):
    #     update_psi_y_and_e(self.kappa_ex, self.sigma_ex, self.a_ex, self.ny, dt, self.dx, self.efield_start, self.efield_end, 
    #                        self.fields.by, self.fields.bz, self.fields.ey, self.fields.ez, self.psi_ey_x, self.psi_ez_x)

    # def advance_b_currents(self, dt):
    #     update_psi_y_and_b(self.kappa_bx, self.sigma_bx, self.a_bx, self.ny, dt, self.dx, self.bfield_start, self.bfield_end, 
    #                        self.fields.ey, self.fields.ez, self.fields.by, self.fields.bz, self.psi_by_x, self.psi_bz_x)

class PMLXmin(PMLX):
    def init_parameters(self):
        # runs from 1.0 to nearly 0.0 (actually 0.0 at cpml_thickness+1)
        pos = 1.0 - np.arange(self.thickness, dtype=float) / self.thickness
        cpml_slice = np.s_[:self.thickness]
        self.init_coefficents(pos, cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 1.0 to nearly 0.0 on the half intervals
        # 1.0 at ix_glob=1-1/2 and 0.0 at ix_glob=cpml_thickness+1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5) / self.thickness
        self.init_coefficents(pos, cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)
        
        # pml range
        self.efield_start = 0
        self.efield_end = self.thickness
        self.bfield_start = 0
        self.bfield_end = self.thickness


class PMLXmax(PMLX):
    def init_parameters(self):
        # runs from nearly 0.0 (actually 0.0 at cpml_thickness+1) to 1.0
        pos = 1.0 - np.arange(self.thickness, dtype=float)[::-1] / self.thickness
        cpml_slice = np.s_[self.nx-self.thickness : self.nx]
        self.init_coefficents(pos, cpml_slice, self.kappa_ex, self.sigma_ex, self.a_ex)

        # runs from nearly 0.0 to nearly 1.0 on the half intervals
        # 0.0 at ix_glob=cpml_thickness+1/2 and 1.0 at ix_glob=1-1/2
        pos = 1.0 - (np.arange(self.thickness, dtype=float) + 0.5)[::-1] / self.thickness
        cpml_slice = np.s_[self.nx-self.thickness-1 : self.nx-1]
        self.init_coefficents(pos, cpml_slice, self.kappa_bx, self.sigma_bx, self.a_bx)

        # pml range
        self.efield_start = self.nx - self.thickness
        self.efield_end = self.nx
        self.bfield_start = self.nx - self.thickness - 1
        self.bfield_end = self.nx - 1


@njit
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
    for j in range(ny):
        bfactor_y = bfactor / kappa_ey[j]
        for i in range(nx):
            bfactor_x = bfactor / kappa_ex[i]
            ex[i, j] += bfactor_y * ( (bz[i, j] - bz[i, j-1]) / dy) - jfactor * jx[i, j]
            ey[i, j] += bfactor_x * (-(bz[i, j] - bz[i-1, j]) / dx) - jfactor * jy[i, j]
            ez[i, j] += bfactor_x * ( (by[i, j] - by[i-1, j]) / dx) \
                      - bfactor_y * ( (bx[i, j] - bx[i, j-1]) / dy) - jfactor * jz[i, j]


@njit
def update_bfield_cpml_2d(
    ex, ey, ez, 
    bx, by, bz, 
    kappa_bx, kappa_by,
    dx, dy, dt, 
    nx, ny, n_guard
):
    for j in range(ny):
        efactor_y = dt / kappa_by[j]
        for i in range(nx):
            efactor_x = dt / kappa_bx[i]
            bx[i, j] -= efactor_y * ( (ez[i, j+1] - ez[i, j]) / dy)
            by[i, j] -= efactor_x * (-(ez[i+1, j] - ez[i, j]) / dx)
            bz[i, j] -= efactor_x * ( (ey[i+1, j] - ey[i, j]) / dx) \
                      - efactor_y * ( (ex[i, j+1] - ex[i, j]) / dy)



@njit(cache=True, parallel=True)
def update_efield_cpml_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    jx_list, jy_list, jz_list,
    kappa_ex_list, kappa_ey_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]
        kappa_ex = kappa_ex_list[ipatch]
        kappa_ey = kappa_ey_list[ipatch]

        update_efield_cpml_2d(ex, ey, ez, bx, by, bz, jx, jy, jz, kappa_ex, kappa_ey, dx, dy, dt, nx, ny, n_guard)


@njit(cache=True, parallel=True)
def update_bfield_cpml_patches_2d(
    ex_list, ey_list, ez_list,
    bx_list, by_list, bz_list,
    kappa_bx_list, kappa_by_list,
    npatches,
    dx, dy, dt,
    nx, ny, n_guard
):
    for ipatch in prange(npatches):
        ex = ex_list[ipatch]
        ey = ey_list[ipatch]
        ez = ez_list[ipatch]
        bx = bx_list[ipatch]
        by = by_list[ipatch]
        bz = bz_list[ipatch]
        kappa_bx = kappa_bx_list[ipatch]
        kappa_by = kappa_by_list[ipatch]

        update_bfield_cpml_2d(ex, ey, ez, bx, by, bz, kappa_bx, kappa_by, dx, dy, dt, nx, ny, n_guard)

@njit
def update_psi_x_and_e(kappa, sigma, a, ny, dt, dx, start, stop, by, bz, ey, ez, psi_ey_x, psi_ez_x):
    fac = dt * c**2
    for iy in range(ny):
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
    for iy in range(ny):
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
