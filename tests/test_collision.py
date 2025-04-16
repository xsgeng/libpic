import unittest
import numpy as np
from libpic.collision.cpu import self_pairing, pairing, self_collision_parallel_2d, inter_collision_parallel_2d,self_collision_cell_2d, self_collision_2d
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u, epsilon_0
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
        self.dt = 0.1e-17
        
        self.m = m_e
        self.q = -e
        
        ppc = 4000
        # npart_in_cell = np.random.randint(100, 150, nx*ny)
        npart_in_cell = np.full(self.nx*self.ny, ppc)
        cell_bound_max = np.cumsum(npart_in_cell)
        cell_bound_min = cell_bound_max - npart_in_cell
        nbuf = npart_in_cell.sum()
        
        self.cell_bound_min = np.reshape(cell_bound_min, (self.nx, self.ny))
        self.cell_bound_max = np.reshape(cell_bound_max, (self.nx, self.ny))
        
        self.dead = np.random.uniform(size=nbuf) < 0.
        
        self.Tex0 = 0.000011
        self.Tey0 = self.Tez0 = 0.00001
        E = gamma(a=3/2, scale=self.Tex0).rvs(nbuf)
        E1 = gamma(a=3/2, scale=self.Tey0).rvs(nbuf)        
        phi = np.arccos(np.random.uniform(-1, 1, nbuf))
        theta = np.random.uniform(0, 2*pi, nbuf)
        beta = np.sqrt(1.-1./(E+1)**2)#np.sqrt(2*E)
        beta1 = np.sqrt(1.-1./(E*self.Tey0/self.Tex0+1)**2)#np.sqrt(2*E1)
        betax = beta * np.cos(theta) * np.sin(phi)
        betay = beta1 * np.sin(theta) * np.sin(phi)
        betaz = beta1 * np.cos(phi)
        self.inv_gamma2 = np.sqrt(1 - (betax**2 + betay**2 + betaz**2))
        self.ux = betax / self.inv_gamma2
        self.uy = betay / self.inv_gamma2 #* np.sqrt(self.Tey0/self.Tex0)
        self.uz = betaz / self.inv_gamma2 #* np.sqrt(self.Tez0/self.Tex0)

        
        self.ne =  1.116e28
        self.w = np.random.uniform(1, 1, nbuf) * self.ne * self.dx * self.dy / ppc
        #self.w = np.ones(nbuf) * self.ne * self.dx * self.dy / ppc

        self.w[self.dead] = 0
        
        self.coulomb_log = 2.0
        self.random_gen = np.random.default_rng()

    def test_self_collision_parallel_2d(self):
      
        test_time = 2e-15
        nsteps = int(test_time / self.dt)
        Tex = np.zeros(nsteps+1)
        Tey = np.zeros(nsteps+1)
        Tez = np.zeros(nsteps+1)
        Tmean = np.zeros(nsteps+1)
        # m_macro = self.m * self.w
        # q_macro = self.q * self.w
        # betax = self.ux * self.inv_gamma2
        # betay = self.uy * self.inv_gamma2
        # betaz = self.uz * self.inv_gamma2
        # Tex[0] = np.mean(betax**2)
        # Tey[0] = np.mean(betay**2)
        # Tez[0] = np.mean(betaz**2)
        # Tmean[0] = (Tex[0] + Tey[0] + Tez[0]) / 3

        ux0 = (self.ux * self.w).sum() / self.w.sum()
        uy0 = (self.uy * self.w).sum() / self.w.sum()
        uz0 = (self.uz * self.w).sum() / self.w.sum()
        Tex_ = ((self.ux-ux0)**2) # mc2
        Tey_ = ((self.uy-uy0)**2) # mc2
        Tez_ = ((self.uz-uz0)**2) # mc2
        Tex[0] = ((Tex_ * self.w).sum() / self.w.sum())
        Tey[0] = ((Tey_ * self.w).sum() / self.w.sum())
        Tez[0] = ((Tez_ * self.w).sum() / self.w.sum())
        Tmean[0] = (Tex[0] + Tey[0] + Tez[0]) / 3

        time = np.arange(nsteps+1) * self.dt
        for _ in range(nsteps):
            self_collision_parallel_2d(
                self.cell_bound_min, self.cell_bound_max, 
                self.nx, self.ny, self.dx, self.dy, self.dt, 
                self.ux, self.uy, self.uz, self.inv_gamma2, self.w, self.dead, 
                self.m, self.q, self.coulomb_log, self.random_gen
            )
            # betax = self.ux * self.inv_gamma2
            # betay = self.uy * self.inv_gamma2
            # betaz = self.uz * self.inv_gamma2
            # Tex[_+1] = np.mean(betax**2)
            # Tey[_+1] = np.mean(betay**2)
            # Tez[_+1] = np.mean(betaz**2)
            # Tmean[_+1] = (Tex[_+1] + Tey[_+1] + Tez[_+1]) / 3

            ux0 = (self.ux * self.w).sum() / self.w.sum()
            uy0 = (self.uy * self.w).sum() / self.w.sum()
            uz0 = (self.uz * self.w).sum() / self.w.sum()
            Tex_ = ((self.ux-ux0)**2) # mc2
            Tey_ = ((self.uy-uy0)**2) # mc2
            Tez_ = ((self.uz-uz0)**2) # mc2
            Tex[_+1] = ((Tex_ * self.w).sum() / self.w.sum())
            Tey[_+1] = ((Tey_ * self.w).sum() / self.w.sum())
            Tez[_+1] = ((Tez_ * self.w).sum() / self.w.sum())
            Tmean[_+1] = (Tex[_+1] + Tey[_+1] + Tez[_+1]) / 3

        Tpar  =Tpar1= Tex[0]
        Tperp =Tperp1= (Tey[0]+Tez[0])/2
        dt_theory = self.dt/100
        t_theory = np.arange(test_time/dt_theory) * dt_theory
        Tpar_theory  = np.zeros_like(t_theory)
        Tperp_theory = np.zeros_like(t_theory)
        Tpar_theory1  = np.zeros_like(t_theory)
        Tperp_theory1 = np.zeros_like(t_theory)        

        re_ = 2.8179403267e-15 # meters
        wavelength =  1e-6 # meters
        coeff = (2*pi/wavelength)**2*re_*c / (2.*np.sqrt(pi))
        density_electron = 10.01027741148843

        for it in range(len(t_theory)):
            Tpar_theory[it] = Tpar
            Tperp_theory[it] = Tperp
            Tpar_theory1[it] = Tpar1
            Tperp_theory1[it] = Tperp1
            A = Tperp/Tpar - 1.
            if A>0: 
                break
            # nu0 = coeff * density_electron * self.coulomb_log /Tpar**1.5 * A**-2 *(
            #     -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )
            
            nu0 = 2.*np.sqrt(pi)*(e**2/(4*pi*epsilon_0))**2*self.ne*self.coulomb_log/(np.sqrt(m_e)*(Tpar*m_e*c**2)**1.5) * A**-2 *(
                -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )# * 1e-6 # 1e-6 to convert to seconds

            A1 = Tpar1/Tperp1 - 1.
            if A1<0: 
                break
            # nu0 = coeff * density_electron * self.coulomb_log /Tpar**1.5 * A**-2 *(
            #     -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )
            
            nu01 = 2.*np.sqrt(pi)*(e**2/(4*pi*epsilon_0))**2*self.ne*self.coulomb_log/(np.sqrt(m_e)*(Tpar*m_e*c**2)**1.5) * A1**-2 *(
                -3. + (A1+3.) * np.arctan((A1)**0.5)/(A1)**0.5 )# * 1e-6 # 1e-6 to convert to seconds
            
            #print A, Tpar, Tperp, nu0
            Tpar  -= 2.*nu0*(Tpar-Tperp)* dt_theory
            Tperp +=    nu0*(Tpar-Tperp)* dt_theory

            Tpar1  -= 2.*nu01*(Tpar1-Tperp1)* dt_theory
            Tperp1 +=    nu01*(Tpar1-Tperp1)* dt_theory            

        #print(self.w[1],self.w[2],self.w[3])
        plt.figure(num=1)
        # plt.plot(t_theory/1e-15, Tpar_theory*0.511e6, label='Tpar_theory', color='black')
        # plt.plot(t_theory/1e-15, Tperp_theory*0.511e6, label='Tperp_theory', color='black')
        plt.plot(t_theory/1e-15, Tpar_theory1*0.511e6, label='Tpar_theory(Tperp_theory)', color='black')
        plt.plot(t_theory/1e-15, Tperp_theory1*0.511e6, color='black')
        plt.plot(time/1e-15, Tex*0.511e6, label='Tpar', color='red')
        plt.plot(time/1e-15, (Tey+Tez)/2*0.511e6, label='Tperp', color='blue')
        #plt.plot(time/1e-15, Tmean*0.511e6, label='Tmean')
        plt.xlabel('Time (fs)')
        plt.ylabel('Temperature (eV)')
        plt.legend()
        plt.show()

        plt.figure(num=2)
        plt.plot(time/1e-15, (Tmean-Tmean[0])/Tmean[0], label='Tmean')
        plt.show()

class TestInterCollision(unittest.TestCase):
    def setUp(self):
        self.nx = self.ny = 10
        self.dx = 0.1e-6
        self.dy = 0.1e-6
        self.dt = 0.1e-17
        
        self.m1 = m_e
        self.q1 = -e
        self.m2 = m_e
        self.Z = 1
        self.q2 = -e*self.Z
        
        ppc1 = 2000
        # npart_in_cell = np.random.randint(100, 150, nx*ny)
        npart_in_cell1 = np.full(self.nx*self.ny, ppc1)
        cell_bound_max1 = np.cumsum(npart_in_cell1)
        cell_bound_min1 = cell_bound_max1 - npart_in_cell1
        nbuf1 = npart_in_cell1.sum()
        self.cell_bound_min1 = np.reshape(cell_bound_min1, (self.nx, self.ny))
        self.cell_bound_max1 = np.reshape(cell_bound_max1, (self.nx, self.ny))

        ppc2 = 2000
        npart_in_cell2 = np.full(self.nx*self.ny, ppc2)
        cell_bound_max2 = np.cumsum(npart_in_cell2)
        cell_bound_min2 = cell_bound_max2 - npart_in_cell2
        nbuf2 = npart_in_cell2.sum()
        self.cell_bound_min2 = np.reshape(cell_bound_min2, (self.nx, self.ny))
        self.cell_bound_max2 = np.reshape(cell_bound_max2, (self.nx, self.ny))
        
        self.dead1 = np.random.uniform(size=nbuf1) < 0.
        self.dead2 = np.random.uniform(size=nbuf2) < 0.

        self.T1 = 0.000011
        self.T2 = 0.00001
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
        self.uy1 = betay1 / self.inv_gamma1 #* np.sqrt(self.T2/self.T1)
        self.uz1 = betaz1 / self.inv_gamma1 #* np.sqrt(self.T2/self.T1)
        self.ux2 = betax2 / self.inv_gamma2
        self.uy2 = betay2 / self.inv_gamma2 #* np.sqrt(self.T2/self.T1)
        self.uz2 = betaz2 / self.inv_gamma2 #* np.sqrt(self.T2/self.T1)

        self.ne1 = 1.116e28
        self.ne2 = 1.116e28
        self.w1 = np.random.uniform(0, 2, nbuf1) * self.ne1 * self.dx * self.dy / ppc1
        self.w2 = np.random.uniform(0, 2, nbuf2) * self.ne2 * self.dx * self.dy / ppc2
        self.w1[self.dead1] = 0
        self.w2[self.dead2] = 0
        
        self.coulomb_log = 2.0
        self.random_gen = np.random.default_rng()
    
    def test_inter_collision_parallel_2d(self):

        test_time = 2e-15
        nsteps = int(test_time / self.dt)
        Tex = np.zeros(nsteps+1)
        Tey = np.zeros(nsteps+1)
        Tez = np.zeros(nsteps+1)

        Tex1 = np.zeros(nsteps+1)
        Tey1 = np.zeros(nsteps+1)
        Tez1 = np.zeros(nsteps+1)
        Tmean1 = np.zeros(nsteps+1)
        # betax1 = self.ux1 * self.inv_gamma1
        # betay1 = self.uy1 * self.inv_gamma1
        # betaz1 = self.uz1 * self.inv_gamma1
        # Tex1[0] = np.mean(betax1**2*self.m1/m_e)
        # Tey1[0] = np.mean(betay1**2*self.m1/m_e)
        # Tez1[0] = np.mean(betaz1**2*self.m1/m_e)
        # Tmean1[0] = (Tex1[0] + Tey1[0] + Tez1[0]) / 3

        ux10 = (self.ux1 * self.w1).sum() / self.w1.sum()
        uy10 = (self.uy1 * self.w1).sum() / self.w1.sum()
        uz10 = (self.uz1 * self.w1).sum() / self.w1.sum()
        Tex1_ = ((self.ux1-ux10)**2)*self.m1/m_e # mc2
        Tey1_ = ((self.uy1-uy10)**2)*self.m1/m_e # mc2
        Tez1_ = ((self.uz1-uz10)**2)*self.m1/m_e # mc2
        Tex1[0] = ((Tex1_ * self.w1).sum() / self.w1.sum())
        Tey1[0] = ((Tey1_ * self.w1).sum() / self.w1.sum())
        Tez1[0] = ((Tez1_ * self.w1).sum() / self.w1.sum())
        Tmean1[0] = (Tex1[0] + Tey1[0] + Tez1[0]) / 3

        Tex2 = np.zeros(nsteps+1)
        Tey2 = np.zeros(nsteps+1)
        Tez2 = np.zeros(nsteps+1)
        Tmean2 = np.zeros(nsteps+1)
        # betax2 = self.ux2 * self.inv_gamma2
        # betay2 = self.uy2 * self.inv_gamma2
        # betaz2 = self.uz2 * self.inv_gamma2
        # Tex2[0] = np.mean(betax2**2*self.m2/m_e)
        # Tey2[0] = np.mean(betay2**2*self.m2/m_e)
        # Tez2[0] = np.mean(betaz2**2*self.m2/m_e)
        # Tmean2[0] = (Tex2[0] + Tey2[0] + Tez2[0]) / 3

        ux20 = (self.ux2 * self.w2).sum() / self.w2.sum()
        uy20 = (self.uy2 * self.w2).sum() / self.w2.sum()
        uz20 = (self.uz2 * self.w2).sum() / self.w2.sum()
        Tex2_ = ((self.ux2-ux20)**2)*self.m2/m_e
        Tey2_ = ((self.uy2-uy20)**2)*self.m2/m_e
        Tez2_ = ((self.uz2-uz20)**2)*self.m2/m_e
        Tex2[0] = ((Tex2_ * self.w2).sum() / self.w2.sum())
        Tey2[0] = ((Tey2_ * self.w2).sum() / self.w2.sum())
        Tez2[0] = ((Tez2_ * self.w2).sum() / self.w2.sum())
        Tmean2[0] = (Tex2[0] + Tey2[0] + Tez2[0]) / 3

        Tex[0] = ((Tex1_ * self.w1).sum() + (Tex2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
        Tey[0] = ((Tey1_ * self.w1).sum() + (Tey2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
        Tez[0] = ((Tez1_ * self.w1).sum() + (Tez2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())

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
            # betax1 = self.ux1 * self.inv_gamma1
            # betay1 = self.uy1 * self.inv_gamma1
            # betaz1 = self.uz1 * self.inv_gamma1
            # Tex1[_+1] = np.mean(betax1**2*self.m1/m_e)
            # Tey1[_+1] = np.mean(betay1**2*self.m1/m_e)
            # Tez1[_+1] = np.mean(betaz1**2*self.m1/m_e)
            # Tmean1[_+1] = (Tex1[_+1] + Tey1[_+1] + Tez1[_+1]) / 3

            # betax2 = self.ux2 * self.inv_gamma2
            # betay2 = self.uy2 * self.inv_gamma2
            # betaz2 = self.uz2 * self.inv_gamma2
            # Tex2[_+1] = np.mean(betax2**2*self.m2/m_e)
            # Tey2[_+1] = np.mean(betay2**2*self.m2/m_e)
            # Tez2[_+1] = np.mean(betaz2**2*self.m2/m_e)
            # Tmean2[_+1] = (Tex2[_+1] + Tey2[_+1] + Tez2[_+1]) / 3

            ux10 = (self.ux1 * self.w1).sum() / self.w1.sum()
            uy10 = (self.uy1 * self.w1).sum() / self.w1.sum()
            uz10 = (self.uz1 * self.w1).sum() / self.w1.sum()
            Tex1_ = ((self.ux1-ux10)**2)*self.m1/m_e # mc2
            Tey1_ = ((self.uy1-uy10)**2)*self.m1/m_e # mc2
            Tez1_ = ((self.uz1-uz10)**2)*self.m1/m_e # mc2
            Tex1[_+1] = ((Tex1_ * self.w1).sum() / self.w1.sum())
            Tey1[_+1] = ((Tey1_ * self.w1).sum() / self.w1.sum())
            Tez1[_+1] = ((Tez1_ * self.w1).sum() / self.w1.sum())
            Tmean1[_+1] = (Tex1[_+1] + Tey1[_+1] + Tez1[_+1]) / 3

            ux20 = (self.ux2 * self.w2).sum() / self.w2.sum()
            uy20 = (self.uy2 * self.w2).sum() / self.w2.sum()
            uz20 = (self.uz2 * self.w2).sum() / self.w2.sum()
            Tex2_ = ((self.ux2-ux20)**2)*self.m2/m_e
            Tey2_ = ((self.uy2-uy20)**2)*self.m2/m_e
            Tez2_ = ((self.uz2-uz20)**2)*self.m2/m_e
            Tex2[_+1] = ((Tex2_ * self.w2).sum() / self.w2.sum())
            Tey2[_+1] = ((Tey2_ * self.w2).sum() / self.w2.sum())
            Tez2[_+1] = ((Tez2_ * self.w2).sum() / self.w2.sum())
            Tmean2[_+1] = (Tex2[_+1] + Tey2[_+1] + Tez2[_+1]) / 3

            Tex[_+1] = ((Tex1_ * self.w1).sum() + (Tex2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
            Tey[_+1] = ((Tey1_ * self.w1).sum() + (Tey2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
            Tez[_+1] = ((Tez1_ * self.w1).sum() + (Tez2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())

        Tpar  = Tex[0]
        Tperp = (Tey[0]+Tez[0])/2
        dt_theory = self.dt/100
        t_theory = np.arange(test_time/dt_theory) * dt_theory
        Tpar_theory  = np.zeros_like(t_theory)
        Tperp_theory = np.zeros_like(t_theory)

        re_ = 2.8179403267e-15 # meters
        wavelength =  1e-6 # meters
        coeff = (2*pi/wavelength)**2*re_*c / (2.*np.sqrt(pi))
        density_electron = 10.01027741148843

        for it in range(len(t_theory)):
            Tpar_theory[it] = Tpar
            Tperp_theory[it] = Tperp
            A = Tperp/Tpar - 1.
            if A>0: 
                break
            nu0 = coeff * density_electron * self.coulomb_log /Tpar**1.5 * A**-2 *(
                -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )
            # nu0 = 2.*np.sqrt(pi)*(e**2/(4*pi*epsilon_0))**2*self.ne*self.coulomb_log/(np.sqrt(m_e)*(Tpar*m_e*c**2)**1.5) * A**-2 *(
            #     -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )# * 1e-6 # 1e-6 to convert to seconds

            #print A, Tpar, Tperp, nu0
            Tpar  -= 2.*nu0*(Tpar-Tperp)* dt_theory
            Tperp +=    nu0*(Tpar-Tperp)* dt_theory

        plt.figure(num=2)
        plt.plot(t_theory/1e-15, Tpar_theory*0.511e6, label='Tpar_theory', color='black')
        plt.plot(t_theory/1e-15, Tperp_theory*0.511e6, label='Tperp_theory', color='black')
        plt.plot(time/1e-15, Tex*0.511e6, label='Tpar', color='red')
        plt.plot(time/1e-15, (Tey+Tez)/2*0.511e6, label='Tperp', color='blue')
        plt.xlabel('Time (fs)')
        plt.ylabel('Temperature (eV)')
        plt.title('Temperature vs Time')
        plt.legend()
        plt.show()


        # T1 = Tmean1[0]
        # T2 = Tmean2[0]
        # dt_theory = self.dt/100
        # t_theory = np.arange(test_time/dt_theory) * dt_theory
        # T1_theory  = np.zeros_like(t_theory)
        # T2_theory = np.zeros_like(t_theory)
        # for it in range(len(t_theory)):
        #     T1_theory[it] = T1
        #     T2_theory[it] = T2       
        #     nu0 = 2./3.*np.sqrt(2./pi) * (e**4*self.Z**2*np.sqrt(self.m1*self.m2)*self.ne2*self.coulomb_log) / (
        #         4.*pi*epsilon_0**2*(self.m1*T1*m_e*c**2 + self.m2*T2*m_e*c**2)**1.5)
        #     T1 -= nu0*(T1-T2)* dt_theory
        #     T2 += nu0*(T1-T2)* dt_theory

        # print(Tex[0], Tey[0], Tez[0])
        # plt.figure(num=2)
        # plt.plot(time/1e-15, Tmean1*0.511e6, label='T1')
        # plt.plot(time/1e-15, Tmean2*0.511e6, label='T2')
        # plt.plot(t_theory/1e-15, T1_theory*0.511e6, label='T1_theory')
        # plt.plot(t_theory/1e-15, T2_theory*0.511e6, label='T2_theory')
        # plt.xlabel('Time (fs)')
        # plt.ylabel('Temperature (eV)')
        # plt.title('Temperature vs Time')
        # plt.legend()
        # plt.show()