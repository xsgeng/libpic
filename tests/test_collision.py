import unittest
import numpy as np
from libpic.collision.cpu import self_pairing, pairing, self_collision_parallel_2d, inter_collision_parallel_2d,self_collision_cell_2d, self_collision_2d
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u
import matplotlib.pyplot as plt

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
        
        self.cell_bound_min = np.reshape(cell_bound_min, (self.nx, self.ny))
        self.cell_bound_max = np.reshape(cell_bound_max, (self.nx, self.ny))
        
        self.dead = np.random.uniform(size=nbuf) < 0.
        
        self.Tex0 = 0.0007
        self.Tey0 = self.Tez0 = 0.0001
        E = gamma(a=3/2, scale=self.Tex0).rvs(nbuf)
        phi = np.arccos(np.random.uniform(-1, 1, nbuf))
        theta = np.random.uniform(0, 2*pi, nbuf)
        beta = np.sqrt(2*E)
        betax = beta * np.cos(theta) * np.sin(phi)
        betay = beta * np.sin(theta) * np.sin(phi)
        betaz = beta * np.cos(phi)
        self.inv_gamma2 = np.sqrt(1 - (betax**2 + betay**2 + betaz**2))
        self.ux = betax / self.inv_gamma2
        self.uy = betay / self.inv_gamma2 * np.sqrt(self.Tey0/self.Tex0)
        self.uz = betaz / self.inv_gamma2 * np.sqrt(self.Tez0/self.Tex0)

        
        self.ne = 1.116e28
        self.w = np.random.uniform(0, 2, nbuf) * self.ne * self.dx * self.dy / ppc
        #self.w = np.ones(nbuf) * self.ne * self.dx * self.dy / ppc

        self.w[self.dead] = 0
        
        self.coulomb_log = 2.0
        self.random_gen = np.random.default_rng()

    def test_self_collision_parallel_2d(self):
      
        test_time = 1e-16
        nsteps = int(test_time / self.dt)
        Tex = np.zeros(nsteps+1)
        Tey = np.zeros(nsteps+1)
        Tez = np.zeros(nsteps+1)
        Tmean = np.zeros(nsteps+1)
        m_macro = self.m * self.w
        q_macro = self.q * self.w
        betax = self.ux * self.inv_gamma2
        betay = self.uy * self.inv_gamma2
        betaz = self.uz * self.inv_gamma2
        Tex[0] = np.mean(betax**2)
        Tey[0] = np.mean(betay**2)
        Tez[0] = np.mean(betaz**2)
        Tmean[0] = (Tex[0] + Tey[0] + Tez[0]) / 3
        time = np.arange(nsteps+1) * self.dt
        for _ in range(nsteps):
            self_collision_parallel_2d(
                self.cell_bound_min, self.cell_bound_max, 
                self.nx, self.ny, self.dx, self.dy, self.dt, 
                self.ux, self.uy, self.uz, self.inv_gamma2, self.w, self.dead, 
                self.m, self.q, self.coulomb_log, self.random_gen
            )
            betax = self.ux * self.inv_gamma2
            betay = self.uy * self.inv_gamma2
            betaz = self.uz * self.inv_gamma2
            Tex[_+1] = np.mean(betax**2)
            Tey[_+1] = np.mean(betay**2)
            Tez[_+1] = np.mean(betaz**2)
            Tmean[_+1] = (Tex[_+1] + Tey[_+1] + Tez[_+1]) / 3
        
        plt.figure(num=1)
        plt.plot(time, Tex, label='Tex')
        plt.plot(time, Tey, label='Tey')
        plt.plot(time, Tez, label='Tez')
        plt.plot(time, Tmean, label='Tmean')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (me*c^2)')
        plt.title('Temperature vs Time')
        plt.legend()
        plt.show()

class TestInterCollision(unittest.TestCase):
    def setUp(self):
        self.nx = self.ny = 10
        self.dx = 0.1e-6
        self.dy = 0.1e-6
        self.dt = 1e-17
        
        self.m1 = m_e
        self.q1 = -e
        self.m2 = m_e
        self.q2 = -e
        
        ppc1 = 1000
        # npart_in_cell = np.random.randint(100, 150, nx*ny)
        npart_in_cell1 = np.full(self.nx*self.ny, ppc1)
        cell_bound_max1 = np.cumsum(npart_in_cell1)
        cell_bound_min1 = cell_bound_max1 - npart_in_cell1
        nbuf1 = npart_in_cell1.sum()
        self.cell_bound_min1 = np.reshape(cell_bound_min1, (self.nx, self.ny))
        self.cell_bound_max1 = np.reshape(cell_bound_max1, (self.nx, self.ny))

        ppc2 = 1000
        npart_in_cell2 = np.full(self.nx*self.ny, ppc2)
        cell_bound_max2 = np.cumsum(npart_in_cell2)
        cell_bound_min2 = cell_bound_max2 - npart_in_cell2
        nbuf2 = npart_in_cell2.sum()
        self.cell_bound_min2 = np.reshape(cell_bound_min2, (self.nx, self.ny))
        self.cell_bound_max2 = np.reshape(cell_bound_max2, (self.nx, self.ny))
        
        self.dead1 = np.random.uniform(size=nbuf1) < 0.
        self.dead2 = np.random.uniform(size=nbuf2) < 0.

        self.T1 = 0.0007
        self.T2 = 0.0001
        E1 = gamma(a=3/2, scale=self.T1).rvs(nbuf1)
        E2 = gamma(a=3/2, scale=self.T2).rvs(nbuf2)
        
        phi1 = np.arccos(np.random.uniform(-1, 1, nbuf1))
        theta1 = np.random.uniform(0, 2*pi, nbuf1)
        phi2 = np.arccos(np.random.uniform(-1, 1, nbuf2))
        theta2 = np.random.uniform(0, 2*pi, nbuf2)

        beta1 = np.sqrt(2*E1*m_e/self.m1)
        beta2 = np.sqrt(2*E2*m_e/self.m2)
        betax1 = beta1 * np.cos(theta1) * np.sin(phi1)
        betay1 = beta1 * np.sin(theta1) * np.sin(phi1)
        betaz1 = beta1 * np.cos(phi1)
        betax2 = beta2 * np.cos(theta2) * np.sin(phi2)
        betay2 = beta2 * np.sin(theta2) * np.sin(phi2)
        betaz2 = beta2 * np.cos(phi2)

        self.inv_gamma1 = np.sqrt(1 - (betax1**2 + betay1**2 + betaz1**2))
        self.inv_gamma2 = np.sqrt(1 - (betax2**2 + betay2**2 + betaz2**2))
        self.ux1 = betax1 / self.inv_gamma1
        self.uy1 = betay1 / self.inv_gamma1
        self.uz1 = betaz1 / self.inv_gamma1
        self.ux2 = betax2 / self.inv_gamma2
        self.uy2 = betay2 / self.inv_gamma2
        self.uz2 = betaz2 / self.inv_gamma2

        self.ne1 = 1.116e28
        self.ne2 = 1.116e28
        self.w1 = np.random.uniform(0, 2, nbuf1) * self.ne1 * self.dx * self.dy / ppc1
        self.w2 = np.random.uniform(0, 2, nbuf2) * self.ne2 * self.dx * self.dy / ppc2
        self.w1[self.dead1] = 0
        self.w2[self.dead2] = 0
        
        self.coulomb_log = 2.0
        self.random_gen = np.random.default_rng()
    
    def test_inter_collision_parallel_2d(self):

        test_time = 1e-15
        nsteps = int(test_time / self.dt)

        Tex1 = np.zeros(nsteps+1)
        Tey1 = np.zeros(nsteps+1)
        Tez1 = np.zeros(nsteps+1)
        Tmean1 = np.zeros(nsteps+1)
        betax1 = self.ux1 * self.inv_gamma1
        betay1 = self.uy1 * self.inv_gamma1
        betaz1 = self.uz1 * self.inv_gamma1
        Tex1[0] = np.mean(betax1**2*self.m1/m_e)
        Tey1[0] = np.mean(betay1**2*self.m1/m_e)
        Tez1[0] = np.mean(betaz1**2*self.m1/m_e)
        Tmean1[0] = (Tex1[0] + Tey1[0] + Tez1[0]) / 3

        Tex2 = np.zeros(nsteps+1)
        Tey2 = np.zeros(nsteps+1)
        Tez2 = np.zeros(nsteps+1)
        Tmean2 = np.zeros(nsteps+1)
        betax2 = self.ux2 * self.inv_gamma2
        betay2 = self.uy2 * self.inv_gamma2
        betaz2 = self.uz2 * self.inv_gamma2
        Tex2[0] = np.mean(betax2**2*self.m2/m_e)
        Tey2[0] = np.mean(betay2**2*self.m2/m_e)
        Tez2[0] = np.mean(betaz2**2*self.m2/m_e)
        Tmean2[0] = (Tex2[0] + Tey2[0] + Tez2[0]) / 3

        time = np.arange(nsteps+1) * self.dt
        for _ in range(nsteps):
            inter_collision_parallel_2d(
                self.cell_bound_min1, self.cell_bound_max1, self.cell_bound_min2, self.cell_bound_max2,
                self.nx, self.ny, self.dx, self.dy, self.dt,
                self.ux1, self.uy1, self.uz1, self.inv_gamma1, self.w1, self.dead1,
                self.ux2, self.uy2, self.uz2, self.inv_gamma2, self.w2, self.dead2,
                self.m1, self.q1, self.m2, self.q2,
                self.coulomb_log, self.random_gen
            )
            betax1 = self.ux1 * self.inv_gamma1
            betay1 = self.uy1 * self.inv_gamma1
            betaz1 = self.uz1 * self.inv_gamma1
            Tex1[_+1] = np.mean(betax1**2*self.m1/m_e)
            Tey1[_+1] = np.mean(betay1**2*self.m1/m_e)
            Tez1[_+1] = np.mean(betaz1**2*self.m1/m_e)
            Tmean1[_+1] = (Tex1[_+1] + Tey1[_+1] + Tez1[_+1]) / 3

            betax2 = self.ux2 * self.inv_gamma2
            betay2 = self.uy2 * self.inv_gamma2
            betaz2 = self.uz2 * self.inv_gamma2
            Tex2[_+1] = np.mean(betax2**2*self.m2/m_e)
            Tey2[_+1] = np.mean(betay2**2*self.m2/m_e)
            Tez2[_+1] = np.mean(betaz2**2*self.m2/m_e)
            Tmean2[_+1] = (Tex2[_+1] + Tey2[_+1] + Tez2[_+1]) / 3
        
        plt.figure(num=2)
        plt.plot(time, Tmean1, label='Tmean1')
        plt.plot(time, Tmean2, label='Tmean2')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (me*c^2)')
        plt.title('Temperature vs Time')
        plt.legend()
        plt.show()