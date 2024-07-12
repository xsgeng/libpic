import unittest
import numpy as np
from libpic.collision.cpu import self_pairing, pairing, self_collision_parallel_2d, inter_collision_parallel_2d


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

    
