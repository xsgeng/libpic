import unittest
import numpy as np
from libpic.collision.cpu import self_pairing, pairing, self_collision_parallel_2d, inter_collision_parallel_2d
from scipy.stats import gamma
from scipy.constants import pi, m_e, e

class TestPairing(unittest.TestCase):
    def setUp(self):
        self.random_gen = np.random.default_rng()

    def test_pairing(self):
        nbuf1 = 100
        nbuf2 = 49
        dead1 = np.random.uniform(size=nbuf1) < 0.5
        dead2 = np.random.uniform(size=nbuf2) < 0.5

        dead1[-1] = True
        dead2[-1] = True

        npart1 = nbuf1 - sum(dead1)
        npart2 = nbuf2 - sum(dead2)

        ip1 = np.zeros(npart1)
        ip2 = np.zeros(npart1)
        for ipair, ip1_, ip2_, w_corr in pairing(dead1, 0, nbuf1, dead2, 0, nbuf2, self.random_gen):
            ip1[ipair] = ip1_
            ip2[ipair] = ip2_
        
        ip1.sort()
        ip1_expected_sorted = np.arange(nbuf1)[np.logical_not(dead1)]

        ip2_expected = np.tile(np.arange(nbuf2)[np.logical_not(dead2)], -(-nbuf1//nbuf2))[:npart1]

        self.assertTrue(np.array_equal(ip1,ip1_expected_sorted))
        self.assertTrue(np.array_equal(ip2,ip2_expected))


    def test_self_pairing(self):
        nbuf = 20
        dead = np.random.uniform(size=nbuf) < 0.5

        dead[-1] = True

        npart = nbuf - sum(dead)
        npairs = (npart + 1) // 2

        ip1 = np.zeros(npairs, dtype=int)
        ip2 = np.zeros(npairs, dtype=int)
        for ipair, ip1_, ip2_, w_corr in self_pairing(dead, 0, nbuf, self.random_gen):
            ip1[ipair] = ip1_
            ip2[ipair] = ip2_

        self.assertTrue((np.union1d(ip1, ip2) == np.arange(nbuf)[np.logical_not(dead)]).all())
        self.assertEqual(np.intersect1d(ip1, ip2).size, npart % 2)
        self.assertFalse((ip1 == ip2).any())

    
class TestSelfCollision(unittest.TestCase):
    def setUp(self):
        self.nx = self.ny = 10
        self.dx = 0.1e-6
        self.dy = 0.1e-6
        self.dt = 1e-17
        
        self.m = m_e
        self.q = -e
        
        ppc = 1000
        # npart_in_cell = np.random.randint(100, 150, nx*ny)
        npart_in_cell = np.full(self.nx*self.ny, ppc)
        cell_bound_max = np.cumsum(npart_in_cell)
        cell_bound_min = cell_bound_max - npart_in_cell
        nbuf = npart_in_cell.sum()
        
        cell_bound_min = np.reshape(cell_bound_min, (self.nx, self.ny))
        cell_bound_max = np.reshape(cell_bound_max, (self.nx, self.ny))
        
        dead = np.random.uniform(size=nbuf) < 0.
        
        self.Tex0 = 0.00011
        self.Tey0 = self.Tez0 = 0.0001
        E = gamma(a=3/2, scale=self.Tex0).rvs(nbuf)
        phi = np.arccos(np.random.uniform(-1, 1, nbuf))
        theta = np.random.uniform(0, 2*pi, nbuf)
        beta = np.sqrt(2*E)
        betax = beta * np.cos(theta) * np.sin(phi)
        betay = beta * np.sin(theta) * np.sin(phi)
        betaz = beta * np.cos(phi)
        inv_gamma2 = np.sqrt(1 - (betax**2 + betay**2 + betaz**2))
        self.ux = betax / inv_gamma2
        self.uy = betay / inv_gamma2 * self.Tey0/self.Tex0
        self.uz = betaz / inv_gamma2 * self.Tez0/self.Tex0

        
        self.ne = 1.116e28
        self.w = np.random.uniform(0, 2, nbuf) * self.ne * self.dx * self.dy / ppc
        self.w[dead] = 0
        
        self.coulomb_log = 2.0