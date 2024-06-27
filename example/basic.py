import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from scipy.constants import c, e, epsilon_0, m_e, mu_0, pi

from libpic.simulation import Simulation
from libpic.species import Electron, Proton

logger.remove()

um = 1e-6
l0 = 0.8 * um
t0 = l0 / c
omega0 = 2 * pi * c / l0
nc = epsilon_0 * m_e * omega0**2 / e**2

nx = 1024
ny = 1024
dx = l0 / 20
dy = l0 / 20

Lx = nx * dx
Ly = ny * dy


def density(x, y):
    ne = 0.0
    if x > Lx / 2 and x < Lx/2 + 4*um:
        ne = 10*nc
    return ne


def add_laser(sim: Simulation):
    a0 = 10
    w0 = 5e-6
    tau = 10

    E0 = a0 * m_e * c * omega0 / e

    for p in sim.patches:
        f = p.fields
        x = f.xaxis
        y = f.yaxis
        f.ey[:] = (
            E0
            * np.exp(-((y - Ly / 2) ** 2) / w0**2)
            * np.sin(omega0 / c * x)
            * np.sin((x - (Lx/2-tau*l0)) / tau / l0 * pi)
        )
        f.ey *= ((x) > Lx/2 - tau*l0) & ((x) < Lx/2)
        f.bz[:] = f.ey / c

bwr_alpha = _LinearSegmentedColormap(
    'bwr_alpha', 
    dict(
        red=[
            (0, 0, 0), 
            (0.5, 1, 1), 
            (1, 1, 1)
        ],
        green=[
            (0, 0.5, 0),
            (0.5, 1, 1),
            (1, 0, 0)
        ],
        blue=[
            (0, 1, 1),
            (0.5, 1, 1),
            (1, 0, 0)
        ],
        alpha = [
            (0, 1, 1),
            (0.5, 0, 0),
            (1, 1, 1)
        ],
    ), 
)

if __name__ == "__main__":
    sim = Simulation(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        npatch_x=16,
        npatch_y=16,
    )

    ele = Electron(density=density, ppc=9)
    ion = Proton(density=density, ppc=2)

    sim.add_species([ele, ion])

    add_laser(sim)

    ne = np.zeros((nx, ny))
    def store_ne(it, ispec):
        if it % 100 == 0 and ispec == 0:
            patches = sim.patches
            nx_per_patch = sim.nx_per_patch
            ny_per_patch = sim.ny_per_patch
            n_guard = sim.n_guard

            for ipatch, p in enumerate(patches):
                s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                          p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                ne[s] = p.fields.rho[:-2*n_guard, :-2*n_guard] / -e / nc

    def plot_ey_rho(it):
        if it % 100 == 0:
            patches = sim.patches
            nx_per_patch = sim.nx_per_patch
            ny_per_patch = sim.ny_per_patch
            n_guard = sim.n_guard

            ey = np.zeros((nx, ny)) 
            for ipatch, p in enumerate(patches):
                s = np.s_[p.ipatch_x*nx_per_patch:p.ipatch_x*nx_per_patch+nx_per_patch,\
                          p.ipatch_y*ny_per_patch:p.ipatch_y*ny_per_patch+ny_per_patch]
                ey[s] = p.fields.ey[:-2*n_guard, :-2*n_guard]
                
            ey *= e / (m_e * c * omega0)

            fig, ax = plt.subplots()
            ax.imshow(
                ne.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap='Grays',
                vmax=20,
                vmin=0,
            )
            ax.imshow(
                ey.T, 
                extent=[0, Lx, 0, Ly],
                origin='lower',
                cmap=bwr_alpha,
                vmax=10,
                vmin=-10,
            )
            fig.savefig(f'ey_{it:04d}.png')

    sim.run(500, callback_species=[store_ne], callback=[plot_ey_rho])
