
from numba import typed

from ..patch import Patches
from ..species import Electron, Photon, Species
from .cpu import (create_photon_patches, photon_recoil_patches,
                  radiation_event_patches, update_chi_patches)


class RadiationBase:
    """
    Radiation class handles creation of photons.

    QED processes are species-wised.
    """
    def __init__(self, patches: Patches, ispec: int) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            The patches.
        ispec : int
            The index of species.
        """
        self.dimension = patches.dimension
        self.patches = patches
        self.npatches = patches.npatches

        self.ispec = ispec


    def generate_particle_lists(self) -> None:
        ispec = self.ispec
        self.tau_list = typed.List([p.particles[ispec].tau for p in self.patches])
        self.chi_list = typed.List([p.particles[ispec].chi for p in self.patches])
        self.delta_list = typed.List([p.particles[ispec].delta for p in self.patches])

        self.ux_list = typed.List([p.particles[ispec].ux for p in self.patches])
        self.uy_list = typed.List([p.particles[ispec].uy for p in self.patches])
        self.uz_list = typed.List([p.particles[ispec].uz for p in self.patches])
        self.inv_gamma_list = typed.List([p.particles[ispec].inv_gamma for p in self.patches])


        self.ex_part_list = typed.List([p.particles[ispec].ex_part for p in self.patches])
        self.ey_part_list = typed.List([p.particles[ispec].ey_part for p in self.patches])
        self.ez_part_list = typed.List([p.particles[ispec].ez_part for p in self.patches])
        self.bx_part_list = typed.List([p.particles[ispec].bx_part for p in self.patches])
        self.by_part_list = typed.List([p.particles[ispec].by_part for p in self.patches])
        self.bz_part_list = typed.List([p.particles[ispec].bz_part for p in self.patches])

        self.event_list = typed.List([p.particles[ispec].event for p in self.patches])
        self.pruned_list = typed.List([p.particles[ispec].pruned for p in self.patches])

    def update_chi(self) -> None:
        update_chi_patches(
            self.ux_list, self.uy_list, self.uz_list,
            self.inv_gamma_list,
            self.ex_part_list, self.ey_part_list, self.ez_part_list,
            self.bx_part_list, self.by_part_list, self.bz_part_list,
            self.pruned_list, self.npatches, self.chi_list,
        )

    def event(self, dt: float) -> None:
        raise NotImplementedError

    def filter(self) -> None:
        raise NotImplementedError

    def create_particles(self) -> None:
        raise NotImplementedError

    def reaction(self) -> None:
        raise NotImplementedError


class NonlinearComptonLCFA(RadiationBase):
    def __init__(self, patches: Patches, ispec: int) -> None:
        super().__init__(patches, ispec)

        radiation_species: Electron = patches.species[ispec]
        assert isinstance(radiation_species, Electron)
        assert isinstance(radiation_species.photon, Photon)

        self.photon_ispec = patches.species.index(radiation_species.photon)

        self.generate_particle_lists()

    def generate_particle_lists(self) -> None:
        super().generate_particle_lists()
        # electrons
        ispec = self.ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_list = typed.List([p.z for p in particles])

        # photons
        ispec = self.photon_ispec
        particles = [p.particles[ispec] for p in self.patches]
        self.x_list = typed.List([p.x for p in particles])
        if self.dimension >= 2:
            self.y_list = typed.List([p.y for p in particles])
        if self.dimension == 3:
            self.z_list = typed.List([p.z for p in particles])
        self.ux_pho_list = typed.List([p.ux for p in particles])
        self.uy_pho_list = typed.List([p.uy for p in particles])
        self.uz_pho_list = typed.List([p.uz for p in particles])
        self.inv_gamma_pho_list = typed.List([p.inv_gamma for p in particles])

        self.pruned_pho_list = typed.List([p.pruned for p in particles])

        self.delta_pho_list = typed.List([p.delta for p in particles])

    def event(self, dt: float) -> None:
        radiation_event_patches(
            self.tau_list, self.chi_list, self.inv_gamma_list,
            self.pruned_list,
            self.npatches, dt, 
            self.event_list, self.delta_list,
        )


    def create_particles(self) -> None:
        # extend photons

        # fillin photons
        create_photon_patches(
            self.x_list, self.y_list, self.z_list, self.ux_list, self.uy_list, self.uz_list,
            self.x_pho_list, self.y_pho_list, self.z_pho_list, self.ux_pho_list, self.uy_pho_list, self.uz_pho_list,
            self.inv_gamma_pho_list, self.pruned_pho_list, self.delta_pho_list,
            self.event_list,
            self.npatches,
        )

    def reaction(self) -> None:
        photon_recoil_patches(
            self.ux_list, self.uy_list, self.uz_list, self.inv_gamma_list,
            self.event_list, self.delta_list, self.pruned_list,
            self.npatches,
        )

class ContinuousRadiation(RadiationBase):

    def __init__(self, patches: Patches, ispec: int) -> None:
        """
        Construct from patches.

        Parameters
        ----------
        patches : Patches
            The patches.
        ispec : int
            The index of species.
        """
        self.patches = patches
        self.npatches = patches.npatches

        self.ispec = ispec

        radiation_species: Electron = patches.species[ispec]
        assert isinstance(radiation_species, Electron)
        assert radiation_species.photon is None

        self.photon_ispec = patches.species.index(radiation_species.photon)

        self.q = radiation_species.q

        self.generate_particle_lists()


    def event(self, dt: float) -> None:
        pass

    def create_particles(self) -> None:
        pass

    def reaction(self) -> None:
        ...