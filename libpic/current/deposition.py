import numpy as np
from numba import njit, prange, typed, types
from scipy.constants import c, e, epsilon_0, mu_0

from libpic.patch import Patches


class CurrentDeposition:
    """
    Current deposition class.

    Holds J, Rho fields and some particle attributes of all patches.

    """
    def __init__(self, patches: Patches) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        
        """
        self.patches = patches
        self.npatches: int = patches.npatches
        self.dx: float = patches.dx

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.w_list = []
        self.ux_list = []
        self.uy_list = []
        self.uz_list = []
        self.inv_gamma_list = []
        self.pruned_list = []

        self.q = []

    def generate_particle_lists(self) -> None:
        """
        Add species to the current deposition class.

        Parameters
        ----------
        particle_list : list of Particles
            List of particles of all patches. 
        """

        for ispec, s in enumerate(self.patches.species):
            self.x_list.append(typed.List([p.particles[ispec].x for p in self.patches]))
            self.w_list.append(typed.List([p.particles[ispec].w for p in self.patches]))
            self.ux_list.append(typed.List([p.particles[ispec].ux for p in self.patches]))
            self.uy_list.append(typed.List([p.particles[ispec].uy for p in self.patches]))
            self.uz_list.append(typed.List([p.particles[ispec].uz for p in self.patches]))
            self.inv_gamma_list.append(typed.List([p.particles[ispec].inv_gamma for p in self.patches]))
            self.pruned_list.append(typed.List([p.particles[ispec].pruned for p in self.patches]))

            self.q.append(s.q)

    def update_particle_lists(self, ipatch: int, ispec: int) -> None:
        """
        Update particle lists of a species in a patch.

        Parameters
        ----------
        ipatch : int
            Patch index.
        ispec : int
            Species index.
        particle : Particles
            Particle object in the patch.
        """
        self.x_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].x
        self.w_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].w
        self.ux_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].ux
        self.uy_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uy
        self.uz_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].uz
        self.inv_gamma_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].inv_gamma
        self.pruned_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].pruned

    
    def generate_field_lists(self) -> None:
        """
        Update field lists of all patches.

        Parameters
        ----------
        fields : list of Fields2D
            List of fields of all patches.
        """
        self.jx_list = typed.List([p.fields.jx for p in self.patches])
        self.jy_list = typed.List([p.fields.jy for p in self.patches])
        self.jz_list = typed.List([p.fields.jz for p in self.patches])
        self.rho_list = typed.List([p.fields.rho for p in self.patches])

        self.x0s = np.array([p.x0 for p in self.patches])

    def update_patches(self) -> None:
        """
        Called when the arangement of patches changes.
        """
        self.generate_field_lists()
        self.generate_particle_lists()
        self.npatches = self.patches.npatches
        # raise NotImplementedError

    def reset(self) -> None:
        """
        Reset J and Rho to zero.
        """
        for ipatch in range(self.npatches):
            self.jx_list[ipatch].fill(0)
            self.jy_list[ipatch].fill(0)
            self.jz_list[ipatch].fill(0)
            self.rho_list[ipatch].fill(0)
        

    def __call__(self, ispec: int, dt: float) -> None:
        """
        Current deposition.

        Parameters
        ----------
        ispec : int
            Species index.
        dt : float
            Time step.
        """
        raise NotImplementedError


class CurrentDeposition2D(CurrentDeposition):
    def __init__(self, patches: Patches) -> None:
        super().__init__(patches)
        self.dy: float = patches.dy


    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        for ispec, s in enumerate(self.patches.species):
            self.y_list.append(typed.List([p.particles[ispec].y for p in self.patches]))


    def update_particle_lists(self, ipatch: int, ispec: int):
        super().update_particle_lists(ipatch, ispec)
        self.y_list[ispec][ipatch] = self.patches[ipatch].particles[ispec].y


    def generate_field_lists(self) -> None:
        super().generate_field_lists()
        self.y0s = np.array([p.y0 for p in self.patches])


    def __call__(self, ispec:int, dt: float) -> None:
        current_deposition_cpu(
            self.rho_list,
            self.jx_list, self.jy_list, self.jz_list,
            self.x0s, self.y0s,
            self.x_list[ispec], self.y_list[ispec], self.ux_list[ispec], self.uy_list[ispec], self.uz_list[ispec],
            self.inv_gamma_list[ispec],
            self.pruned_list[ispec],
            self.npatches,
            self.dx, self.dy, dt, self.w_list[ispec], self.q[ispec],
        )


@njit(cache=True, parallel=True)
def current_deposition_cpu(
    rho_list,
    jx_list, jy_list, jz_list,
    x0_list, y0_list,
    x_list, y_list, ux_list, uy_list, uz_list,
    inv_gamma_list,
    pruned_list,
    npatches,
    dx, dy, dt, w_list, q,
) -> None:
    """
    Current deposition on all patches in 2D for CPU.

    *_list are the data in all patches.

    Parameters
    ----------
    rho : 2D array
        Charge density.
    jx, jy, jz : 2D arrays
        Current density in x, y, z directions.
    x0, y0 : floats
        Start position of the patche.
    x, y : 1D arrays
        Particle positions.
    uz : 1D array
        Momentum z.
    inv_gamma : 1D array
        Particle inverse gamma factor.
    pruned : 1D array of booleans
        Boolean array indicating if the particle has been pruned.
    npatches : int
        Number of patches.
    dx, dy : floats
        Cell sizes in x and y directions.
    dt : float
        Time step.
    w : 1D array of floats
        Particle weights.
    q : float
        Charge of the particles.
    """
    for ipatch in prange(npatches):
        rho = rho_list[ipatch]
        jx = jx_list[ipatch]
        jy = jy_list[ipatch]
        jz = jz_list[ipatch]
        x0 = x0_list[ipatch]
        y0 = y0_list[ipatch]
        x = x_list[ipatch]
        y = y_list[ipatch]
        ux = ux_list[ipatch]
        uy = uy_list[ipatch]
        uz = uz_list[ipatch]
        w = w_list[ipatch]
        inv_gamma = inv_gamma_list[ipatch]
        pruned = pruned_list[ipatch]
        npart = len(pruned)

        current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q)

class CurrentDeposition3D(CurrentDeposition):
    ...


nbuff = 64

@njit(boundscheck=True)
def current_deposit_2d(rho, jx, jy, jz, x, y, ux, uy, uz, inv_gamma, pruned, npart, dx, dy, x0, y0, dt, w, q):
    """
    Current deposition in 2D for CPU.

    Parameters
    ----------
    rho : 2D array of floats
        Charge density.
    jx, jy, jz : 2D arrays of floats
        Current density in x, y, z directions.
    x, y : 1D arrays of floats
        Particle positions.
    uz : 1D array of floats
        Particle velocities.
    inv_gamma : 1D array of floats
        Particle inverse gamma.
    pruned : 1D array of booleans
        Boolean array indicating if the particle has been pruned.
    npart : int
        Number of particles.
    dx, dy : floats
        Cell sizes in x and y directions.
    dt : float
        Time step.
    w : 1D array of floats
        Particle weights.
    q : float
        Charge of the particles.
    """

    x_old = np.zeros(nbuff) 
    y_old = np.zeros(nbuff) 
    x_adv = np.zeros(nbuff) 
    y_adv = np.zeros(nbuff) 
    vz = np.zeros(nbuff) 
   
    S0x = np.zeros((5, nbuff))
    S1x = np.zeros((5, nbuff))
    S0y = np.zeros((5, nbuff))
    S1y = np.zeros((5, nbuff))
    DSx = np.zeros((5, nbuff))
    DSy = np.zeros((5, nbuff))
    jy_buff = np.zeros((5, nbuff))
    
    for ibuff in range(0, npart, nbuff):
        npart_buff = min(nbuff, npart - ibuff)
        for ip in range(npart_buff):
            ipart_global = ibuff + ip
            if pruned[ipart_global]:
                vz[ip] = 0.0
                x_old[ip] = 0.0
                y_old[ip] = 0.0
                x_adv[ip] = 0.0
                y_adv[ip] = 0.0
                continue
            vx = ux[ipart_global]*c*inv_gamma[ipart_global]
            vy = uy[ipart_global]*c*inv_gamma[ipart_global]
            vz[ip] = uz[ipart_global]*c*inv_gamma[ipart_global] if ~pruned[ipart_global] else 0.0
            x_old[ip] = x[ipart_global] - vx*0.5*dt - x0
            y_old[ip] = y[ipart_global] - vy*0.5*dt - y0
            x_adv[ip] = x[ipart_global] + vx*0.5*dt - x0
            y_adv[ip] = y[ipart_global] + vy*0.5*dt - y0

        for ip in range(npart_buff):
            ipart_global = ibuff + ip
            # positions at t + dt/2, before pusher
            # +0.5 for cell-centered coordinate
            x_over_dx0 = x_old[ip] / dx
            ix0 = int(np.floor(x_over_dx0+0.5))
            y_over_dy0 = y_old[ip] / dy
            iy0 = int(np.floor(y_over_dy0+0.5))

            calculate_S(x_over_dx0 - ix0, 0, ip, S0x)
            calculate_S(y_over_dy0 - iy0, 0, ip, S0y)

            # positions at t + 3/2*dt, after pusher
            x_over_dx1 = x_adv[ip] / dx
            ix1 = int(np.floor(x_over_dx1+0.5))
            dcell_x = ix1 - ix0

            y_over_dy1 = y_adv[ip] / dy
            iy1 = int(np.floor(y_over_dy1+0.5))
            dcell_y = iy1 - iy0

            calculate_S(x_over_dx1 - ix1, dcell_x, ip, S1x)
            calculate_S(y_over_dy1 - iy1, dcell_y, ip, S1y)

            for i in range(5):
                DSx[i, ip] = S1x[i, ip] - S0x[i, ip]
                DSy[i, ip] = S1y[i, ip] - S0y[i, ip]
                jy_buff[i, ip] = 0

            one_third = 1.0 / 3.0
            charge_density = q * w[ipart_global] / (dx*dy) 
            charge_density *= ~pruned[ipart_global]
            factor = charge_density / dt
            

            # i and j are the relative shift, 0-based index
            # [0,   1, 2, 3, 4]
            #     [-1, 0, 1, 2] for dcell = 1;
            #     [-1, 0, 1] for dcell_ = 0
            # [-2, -1, 0, 1] for dcell = -1
            for j in range(min(1, 1+dcell_y), max(4, 4+dcell_y)):
                jx_buff = 0.0
                iy = iy0 + (j - 2)
                for i in range(min(1, 1+dcell_x), max(4, 4+dcell_x)):
                    ix = ix0 + (i - 2)

                    wx = DSx[i, ip] * (S0y[j, ip] + 0.5 * DSy[j, ip])
                    wy = DSy[j, ip] * (S0x[i, ip] + 0.5 * DSx[i, ip])
                    wz = S0x[i, ip] * S0y[j, ip] + 0.5 * DSx[i, ip] * S0y[j, ip] \
                        + 0.5 * S0x[i, ip] * DSy[j, ip] + one_third * DSx[i, ip] * DSy[j, ip]
                    
                    jx_buff -= factor * dx * wx
                    jy_buff[i, ip] -= factor * dy * wy

                    jx[ix, iy] += jx_buff
                    jy[ix, iy] += jy_buff[i, ip]
                    jz[ix, iy] += factor * wz * vz[ip]
                    rho[ix, iy] += charge_density * S1x[i, ip] * S1y[j, ip]


@njit(inline="always")
def calculate_S(delta, shift, ip, S):
    delta2 = delta * delta

    delta_minus    = 0.5 * ( delta2+delta+0.25 )
    delta_mid      = 0.75 - delta2
    delta_positive = 0.5 * ( delta2-delta+0.25 )

    minus = shift == -1
    mid = shift == 0
    positive = shift == 1

    S[0, ip] = minus * delta_minus
    S[1, ip] = minus * delta_mid      + mid * delta_minus
    S[2, ip] = minus * delta_positive + mid * delta_mid      + positive * delta_minus
    S[3, ip] =                          mid * delta_positive + positive * delta_mid
    S[4, ip] =                                                 positive * delta_positive

