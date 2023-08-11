
from numba import njit


@njit(cache=True)
def update_photon_gamma( ux, uy, uz, inv_gamma, npart, pruned) :
    for ip in range(npart):
        if pruned[ip]:
            continue

        inv_gamma[ip] = (ux[ip]*ux[ip] +uy[ip]*uy[ip] + uz[ip]*uz[ip]) ** -0.5