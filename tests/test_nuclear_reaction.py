import unittest
import numpy as np
from libpic.collision.nuclear_reaction import  DDfusion_2d, FWHM
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u, epsilon_0
from numba.typed import List
from numba import float64
from scipy.stats import norm
import matplotlib.pyplot as plt

class TestDDfusion(unittest.TestCase):
    def setUp(self):
        self.nx = self.ny = 20
        self.dx = 100e-6
        self.dy = 100e-6
        self.dt = 50e-15
        
        self.m1 = 2.014102*m_u
        self.q1 = 2.*e
        self.m2 = 2.014102*m_u
        self.Z = 2.
        self.q2 = e*self.Z
        self.pd_m1 = 5497.9*m_e
        self.pd_q1 = 2.*e
        self.pd_m2 = 1838.7*m_e
        self.pd_q2 = 0      
        self.ppc1 = 10000
        # npart_in_cell = np.random.randint(100, 150, nx*ny)
        npart_in_cell1 = np.full(self.nx*self.ny, self.ppc1)
        cell_bound_max1 = np.cumsum(npart_in_cell1)
        cell_bound_min1 = cell_bound_max1 - npart_in_cell1
        nbuf1 = npart_in_cell1.sum()
        self.cell_bound_min1 = np.reshape(cell_bound_min1, (self.nx, self.ny))
        self.cell_bound_max1 = np.reshape(cell_bound_max1, (self.nx, self.ny))

        self.ppc2 = 10000
        npart_in_cell2 = np.full(self.nx*self.ny, self.ppc2)
        cell_bound_max2 = np.cumsum(npart_in_cell2)
        cell_bound_min2 = cell_bound_max2 - npart_in_cell2
        nbuf2 = npart_in_cell2.sum()
        self.cell_bound_min2 = np.reshape(cell_bound_min2, (self.nx, self.ny))
        self.cell_bound_max2 = np.reshape(cell_bound_max2, (self.nx, self.ny))
        
        self.dead1 = np.random.uniform(size=nbuf1) < 0.
        self.dead2 = np.random.uniform(size=nbuf2) < 0.
        self.Tion = 50. #keV
        self.T1 = self.Tion/511.
        #self.T2 = 0.00001
        E1 = gamma(a=3/2, scale=self.T1).rvs(nbuf1)
        #E2 = gamma(a=3/2, scale=self.T2).rvs(nbuf2)
        
        phi1 = np.arccos(np.random.uniform(-1, 1, nbuf1))
        theta1 = np.random.uniform(0, 2*pi, nbuf1)
        phi2 = np.arccos(np.random.uniform(-1, 1, nbuf2))
        theta2 = np.random.uniform(0, 2*pi, nbuf2)

        beta1 = np.sqrt(2*E1*m_e/self.m1)
        beta2 = np.sqrt(2*E1*m_e/self.m2)
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

        self.ne1 = 1.0e26
        self.ne2 = 1.0e26
        self.w1 = np.random.uniform(0, 2, nbuf1) * self.ne1 * self.dx * self.dy / self.ppc1
        self.w2 = np.random.uniform(0, 2, nbuf2) * self.ne2 * self.dx * self.dy / self.ppc2
        self.w1[self.dead1] = 0
        self.w2[self.dead2] = 0
        
        self.coulomb_log = 2.0  ##?
        self.random_gen = np.random.default_rng() 

    def test_inter_collision_parallel_2d(self):

        test_time = 1e-12
        rate_multiplier_ = 1.0e8
        #iproduct = 0
        tot_probability_ = np.array([0.0])
        pd_idx_list = np.array([0])
        nsteps = int(test_time / self.dt)
        # probtest = np.zeros(1000)
        # ekin_test = np.zeros(1000)
        pd_ux1 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_uy1 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_uz1 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_inv_gamma1 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_w1 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_dead1 = np.random.uniform(size=(nsteps+1)*self.ppc1*self.nx*self.ny) > -1.0
        pd_ux2 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_uy2 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_uz2 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_inv_gamma2 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_w2 = np.zeros((nsteps+1)*self.ppc1*self.nx*self.ny)
        pd_dead2 = np.random.uniform(size=(nsteps+1)*self.ppc1*self.nx*self.ny) > -1.0

        Tex = np.zeros(nsteps+1)
        Tey = np.zeros(nsteps+1)
        Tez = np.zeros(nsteps+1)

        ux10 = (self.ux1 * self.w1).sum() / self.w1.sum()
        uy10 = (self.uy1 * self.w1).sum() / self.w1.sum()
        uz10 = (self.uz1 * self.w1).sum() / self.w1.sum()
        Tex1_ = ((self.ux1-ux10)**2)*self.m1/m_e # mc2
        Tey1_ = ((self.uy1-uy10)**2)*self.m1/m_e # mc2
        Tez1_ = ((self.uz1-uz10)**2)*self.m1/m_e # mc2

        ux20 = (self.ux2 * self.w2).sum() / self.w2.sum()
        uy20 = (self.uy2 * self.w2).sum() / self.w2.sum()
        uz20 = (self.uz2 * self.w2).sum() / self.w2.sum()
        Tex2_ = ((self.ux2-ux20)**2)*self.m2/m_e
        Tey2_ = ((self.uy2-uy20)**2)*self.m2/m_e
        Tez2_ = ((self.uz2-uz20)**2)*self.m2/m_e

        Tex[0] = ((Tex1_ * self.w1).sum() + (Tex2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
        Tey[0] = ((Tey1_ * self.w1).sum() + (Tey2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
        Tez[0] = ((Tez1_ * self.w1).sum() + (Tez2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())

        time = np.arange(nsteps+1) * self.dt
        for _ in range(nsteps):
            DDfusion_2d(
                self.cell_bound_min1, self.cell_bound_max1, self.cell_bound_min2, self.cell_bound_max2,
                self.nx, self.ny, self.dx, self.dy, self.dt,
                self.ux1, self.uy1, self.uz1, self.inv_gamma1, self.w1, self.dead1,
                self.ux2, self.uy2, self.uz2, self.inv_gamma2, self.w2, self.dead2,
                self.m1, self.q1, self.m2, self.q2,
                pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
                pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
                self.pd_m1, self.pd_q1, self.pd_m2, self.pd_q2,
                self.coulomb_log, self.random_gen, rate_multiplier_, tot_probability_, pd_idx_list #, iproduct, probtest, ekin_test
            )           
            ux10 = (self.ux1 * self.w1).sum() / self.w1.sum()
            uy10 = (self.uy1 * self.w1).sum() / self.w1.sum()
            uz10 = (self.uz1 * self.w1).sum() / self.w1.sum()
            Tex1_ = ((self.ux1-ux10)**2)*self.m1/m_e # mc2
            Tey1_ = ((self.uy1-uy10)**2)*self.m1/m_e # mc2
            Tez1_ = ((self.uz1-uz10)**2)*self.m1/m_e # mc2

            ux20 = (self.ux2 * self.w2).sum() / self.w2.sum()
            uy20 = (self.uy2 * self.w2).sum() / self.w2.sum()
            uz20 = (self.uz2 * self.w2).sum() / self.w2.sum()
            Tex2_ = ((self.ux2-ux20)**2)*self.m2/m_e
            Tey2_ = ((self.uy2-uy20)**2)*self.m2/m_e
            Tez2_ = ((self.uz2-uz20)**2)*self.m2/m_e

            Tex[_+1] = ((Tex1_ * self.w1).sum() + (Tex2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
            Tey[_+1] = ((Tey1_ * self.w1).sum() + (Tey2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
            Tez[_+1] = ((Tez1_ * self.w1).sum() + (Tez2_ * self.w2).sum()) / (self.w1.sum()+self.w2.sum())
            Tmean = (Tex + Tey + Tez) / 3

    
        pd_idx = np.where(pd_dead2 == True)[0][0]
        pd_ux2 = pd_ux2[:pd_idx]
        pd_uy2 = pd_uy2[:pd_idx]
        pd_uz2 = pd_uz2[:pd_idx]
        pd_w2 = pd_w2[:pd_idx]
        pd_dead2 = pd_dead2[:pd_idx]
        pd_inv_gamma2 = pd_inv_gamma2[:pd_idx]
        unx0 = (pd_ux2 * pd_w2).sum() / pd_w2.sum()
        uny0 = (pd_uy2 * pd_w2).sum() / pd_w2.sum()
        unz0 = (pd_uz2 * pd_w2).sum() / pd_w2.sum()
        Tnex = ((pd_ux2-unx0)**2*pd_w2).sum() / pd_w2.sum()*self.pd_m2/m_e
        Tney = ((pd_uy2-uny0)**2*pd_w2).sum() / pd_w2.sum()*self.pd_m2/m_e
        Tnez = ((pd_uz2-unz0)**2*pd_w2).sum() / pd_w2.sum()*self.pd_m2/m_e
        Tn_tol = Tnex + Tney + Tnez
        Kn = (1/pd_inv_gamma2-1)*self.pd_m2*c**2/e
        csv = pd_w2.sum()*2./(test_time*self.ne1**2*4.*self.dx*self.dy*self.nx*self.ny)
        paramsDD_FWHM = np.array([82.542, 1.7013e-3, 0.16888, 0.49, 7.9460e-4, 8.4619e-3, 8.3241e-4])
        FWHM1 = FWHM(self.Tion, paramsDD_FWHM)
        Kn_mean = (Kn*pd_w2).sum() / pd_w2.sum()
        Kn_var = ((Kn-Kn_mean)**2*pd_w2).sum() / pd_w2.sum()
        FWHM2 = np.sqrt(8*np.log(2)*Kn_var)/1000.  #keV

        print(np.size(pd_ux2))
        print(pd_idx_list)
        #print(pd_dead2[0])
        print(pd_w2.sum())
        print(tot_probability_)
        print(csv)
        print(FWHM1, FWHM2)
        #plt.figure(num=1)        
        plt.plot(time/1e-15, Tmean*0.511e6, label='Tmean')
        plt.plot(time/1e-15, Tex*0.511e6, label='Tex')
        plt.plot(time/1e-15, Tey*0.511e6, label='Tey')
        plt.plot(time/1e-15, Tez*0.511e6, label='Tez')
        #plt.plot(time/1e-15, Tmean2*0.511e6, label='T2')
        plt.xlabel('Time (fs)')
        plt.ylabel('Temperature (eV)')
        plt.title('Temperature vs Time')
        plt.legend()
        plt.show()

        # plt.figure(num=2)
        # plt.plot(ekin_test,probtest, 'o')
        # plt.show()

        plt.figure(num=3)
        plt.hist(Kn/1.0e6, bins=100, weights=pd_w2, density=True, color='blue', edgecolor='black')
        mean, std_dev = norm.fit(Kn/1.0e6)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = norm.pdf(x, mean, std_dev)
        plt.plot(x, pdf, 'r-', label=f'Fitted Normal Distribution\nMean={mean:.2f}, Std={std_dev:.2f}')
        plt.xlabel('En(MeV)')
        plt.ylabel('Probability Density')
        plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 数据
# data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])  # 数据点
# weights = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])  # 每个数据点的权重

# # 绘制直方图
# plt.hist(data, bins=np.arange(1, 6), weights=weights, edgecolor='black', alpha=0.7)

# # 添加标签
# plt.xlabel('Value')
# plt.ylabel('Weighted Frequency')
# plt.title('Histogram with Weights')

# # 显示图形
# plt.show()