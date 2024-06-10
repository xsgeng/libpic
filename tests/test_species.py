from libpic.species import Species, Electron, Photon
from libpic.particles import ParticlesBase, SpinParticles, QEDParticles, SpinQEDParticles
from unittest import TestCase


class TestSpecies(TestCase):
    def test_species(self):
        e = Electron()
        self.assertIsNone(e.radiation)

        e = Electron(polarization=(1, 0, 0))
        p = e.create_particles()
        self.assertIsInstance(p, SpinParticles)

        pho = Photon()
        with self.assertRaises(Exception):
            e.set_photon(pho)
        
        e.radiation = "photons"
        e.set_photon(pho)

        p = e.create_particles()


        

        self.assertIsInstance(p, QEDParticles)
        self.assertIsInstance(p, SpinQEDParticles)