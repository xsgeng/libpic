import pandas as pd
import numpy as np
from libpic.collision.nuclear_reaction import  DDfusion_2d, reaction_rate, FWHM
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u, epsilon_0
from numba.typed import List
from numba import float64
from scipy.stats import norm
import matplotlib.pyplot as plt

nx = ny = 20
dx = 100e-6
dy = 100e-6
#dt = 50e-15

m1 = 2.014102*m_u
q1 = 2.*e
m2 = 2.014102*m_u
Z = 2.
q2 = e*Z
pd_m1 = 5497.9*m_e
pd_q1 = 2.*e
pd_m2 = 1838.7*m_e
pd_q2 = 0      
ppc1 = 10000
# npart_in_cell = np.random.randint(100, 150, nx*ny)
npart_in_cell1 = np.full(nx*ny, ppc1)
cell_bound_max1 = np.cumsum(npart_in_cell1)
cell_bound_min1 = cell_bound_max1 - npart_in_cell1
nbuf1 = npart_in_cell1.sum()
cell_bound_min1 = np.reshape(cell_bound_min1, (nx, ny))
cell_bound_max1 = np.reshape(cell_bound_max1, (nx, ny))

ppc2 = 10000
npart_in_cell2 = np.full(nx*ny, ppc2)
cell_bound_max2 = np.cumsum(npart_in_cell2)
cell_bound_min2 = cell_bound_max2 - npart_in_cell2
nbuf2 = npart_in_cell2.sum()
cell_bound_min2 = np.reshape(cell_bound_min2, (nx, ny))
cell_bound_max2 = np.reshape(cell_bound_max2, (nx, ny))

dead1 = np.random.uniform(size=nbuf1) < 0.
dead2 = np.random.uniform(size=nbuf2) < 0.

Tion_list = np.array([5.0, 10.0, 20.0, 40.0 ])
csv = np.zeros_like(Tion_list)
FWHM_sim = np.zeros_like(Tion_list)

t_step = np.array([ 25e-15, 50e-15, 100e-15, 500e-15])
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.figure(num=1)
#i =0
#for Tion in Tion_list:
for i, Tion in enumerate(Tion_list):
    dt = 50e-15
    T1 = Tion/511.
    #T2 = 0.00001
    E1 = gamma(a=3/2, scale=T1).rvs(nbuf1)
    #E2 = gamma(a=3/2, scale=T2).rvs(nbuf2)

    phi1 = np.arccos(np.random.uniform(-1, 1, nbuf1))
    theta1 = np.random.uniform(0, 2*pi, nbuf1)
    phi2 = np.arccos(np.random.uniform(-1, 1, nbuf2))
    theta2 = np.random.uniform(0, 2*pi, nbuf2)

    beta1 = np.sqrt(2*E1*m_e/m1)
    beta2 = np.sqrt(2*E1*m_e/m2)
    betax1 = beta1 * np.cos(theta1) * np.sin(phi1)
    betay1 = beta1 * np.sin(theta1) * np.sin(phi1)
    betaz1 = beta1 * np.cos(phi1)
    betax2 = beta2 * np.cos(theta2) * np.sin(phi2)
    betay2 = beta2 * np.sin(theta2) * np.sin(phi2)
    betaz2 = beta2 * np.cos(phi2)

    inv_gamma1 = np.sqrt(1 - (betax1**2 + betay1**2 + betaz1**2))
    inv_gamma2 = np.sqrt(1 - (betax2**2 + betay2**2 + betaz2**2))
    ux1 = betax1 / inv_gamma1
    uy1 = betay1 / inv_gamma1 #* np.sqrt(T2/T1)
    uz1 = betaz1 / inv_gamma1 #* np.sqrt(T2/T1)
    ux2 = betax2 / inv_gamma2
    uy2 = betay2 / inv_gamma2 #* np.sqrt(T2/T1)
    uz2 = betaz2 / inv_gamma2 #* np.sqrt(T2/T1)

    ne1 = 1.0e26
    ne2 = 1.0e26
    w1 = np.random.uniform(0, 2, nbuf1) * ne1 * dx * dy / ppc1
    w2 = np.random.uniform(0, 2, nbuf2) * ne2 * dx * dy / ppc2
    w1[dead1] = 0
    w2[dead2] = 0

    coulomb_log = 2.0  ##?
    random_gen = np.random.default_rng() 


    test_time = 5e-12
    rate_multiplier_ = 1.0e9
    #iproduct = 0
    tot_probability_ = np.array([0.0])
    pd_idx_list = np.array([0])
    nsteps = int(test_time / dt)
    # probtest = np.zeros(1000)
    # ekin_test = np.zeros(1000)
    pd_ux1 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_uy1 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_uz1 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_inv_gamma1 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_w1 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_dead1 = np.random.uniform(size=(nsteps+1)*ppc1*nx*ny) > -1.0
    pd_ux2 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_uy2 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_uz2 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_inv_gamma2 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_w2 = np.zeros((nsteps+1)*ppc1*nx*ny)
    pd_dead2 = np.random.uniform(size=(nsteps+1)*ppc1*nx*ny) > -1.0

    Tex = np.zeros(nsteps+1)
    Tey = np.zeros(nsteps+1)
    Tez = np.zeros(nsteps+1)

    ux10 = (ux1 * w1).sum() / w1.sum()
    uy10 = (uy1 * w1).sum() / w1.sum()
    uz10 = (uz1 * w1).sum() / w1.sum()
    Tex1_ = ((ux1-ux10)**2)*m1/m_e # mc2
    Tey1_ = ((uy1-uy10)**2)*m1/m_e # mc2
    Tez1_ = ((uz1-uz10)**2)*m1/m_e # mc2

    ux20 = (ux2 * w2).sum() / w2.sum()
    uy20 = (uy2 * w2).sum() / w2.sum()
    uz20 = (uz2 * w2).sum() / w2.sum()
    Tex2_ = ((ux2-ux20)**2)*m2/m_e
    Tey2_ = ((uy2-uy20)**2)*m2/m_e
    Tez2_ = ((uz2-uz20)**2)*m2/m_e

    Tex[0] = ((Tex1_ * w1).sum() + (Tex2_ * w2).sum()) / (w1.sum()+w2.sum())
    Tey[0] = ((Tey1_ * w1).sum() + (Tey2_ * w2).sum()) / (w1.sum()+w2.sum())
    Tez[0] = ((Tez1_ * w1).sum() + (Tez2_ * w2).sum()) / (w1.sum()+w2.sum())

    time = np.arange(nsteps+1) * dt
    for _ in range(nsteps):
        DDfusion_2d(
            cell_bound_min1, cell_bound_max1, cell_bound_min2, cell_bound_max2,
            nx, ny, dx, dy, dt,
            ux1, uy1, uz1, inv_gamma1, w1, dead1,
            ux2, uy2, uz2, inv_gamma2, w2, dead2,
            m1, q1, m2, q2,
            pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
            pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
            pd_m1, pd_q1, pd_m2, pd_q2,
            coulomb_log, random_gen, rate_multiplier_, tot_probability_, pd_idx_list #, iproduct, probtest, ekin_test
        )           
        ux10 = (ux1 * w1).sum() / w1.sum()
        uy10 = (uy1 * w1).sum() / w1.sum()
        uz10 = (uz1 * w1).sum() / w1.sum()
        Tex1_ = ((ux1-ux10)**2)*m1/m_e # mc2
        Tey1_ = ((uy1-uy10)**2)*m1/m_e # mc2
        Tez1_ = ((uz1-uz10)**2)*m1/m_e # mc2

        ux20 = (ux2 * w2).sum() / w2.sum()
        uy20 = (uy2 * w2).sum() / w2.sum()
        uz20 = (uz2 * w2).sum() / w2.sum()
        Tex2_ = ((ux2-ux20)**2)*m2/m_e
        Tey2_ = ((uy2-uy20)**2)*m2/m_e
        Tez2_ = ((uz2-uz20)**2)*m2/m_e

        Tex[_+1] = ((Tex1_ * w1).sum() + (Tex2_ * w2).sum()) / (w1.sum()+w2.sum())
        Tey[_+1] = ((Tey1_ * w1).sum() + (Tey2_ * w2).sum()) / (w1.sum()+w2.sum())
        Tez[_+1] = ((Tez1_ * w1).sum() + (Tez2_ * w2).sum()) / (w1.sum()+w2.sum())
        Tmean = (Tex + Tey + Tez) / 3


    pd_idx = np.where(pd_dead2 == True)[0][0]
    pd_ux2 = pd_ux2[:pd_idx]
    pd_uy2 = pd_uy2[:pd_idx]
    pd_uz2 = pd_uz2[:pd_idx]
    pd_w2 = pd_w2[:pd_idx]
    pd_dead2 = pd_dead2[:pd_idx]
    pd_inv_gamma2 = pd_inv_gamma2[:pd_idx]
    # unx0 = (pd_ux2 * pd_w2).sum() / pd_w2.sum()
    # uny0 = (pd_uy2 * pd_w2).sum() / pd_w2.sum()
    # unz0 = (pd_uz2 * pd_w2).sum() / pd_w2.sum()
    # Tnex = ((pd_ux2-unx0)**2*pd_w2).sum() / pd_w2.sum()*pd_m2/m_e
    # Tney = ((pd_uy2-uny0)**2*pd_w2).sum() / pd_w2.sum()*pd_m2/m_e
    # Tnez = ((pd_uz2-unz0)**2*pd_w2).sum() / pd_w2.sum()*pd_m2/m_e
    # Tn_tol = Tnex + Tney + Tnez
    Kn = (1/pd_inv_gamma2-1)*pd_m2*c**2/e
    csv[i] = pd_w2.sum()*2./(test_time*ne1**2*4.*dx*dy*nx*ny)
    Kn_mean = (Kn*pd_w2).sum() / pd_w2.sum()
    Kn_var = ((Kn-Kn_mean)**2*pd_w2).sum() / pd_w2.sum()
    FWHM_sim[i] = np.sqrt(8*np.log(2)*Kn_var)/1000.  #keV
    #i += 1
    if i == 0:

        plt.figure(num=1)
        plt.hist(Kn/1.0e6, bins=100, weights=pd_w2, density=True, color='white', edgecolor='white')
    plt.figure(num=1)
    mean, std_dev = norm.fit(Kn/1.0e6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mean, std_dev)
    plt.plot(x, pdf,label=f'Tion={Tion}keV', color=colors[i % len(colors)])

Tion_theory = np.linspace(0., 21, 1000)
paramsDD_csv = np.array([5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0, -2.96400e-6, 0.0, 0.0, 31.3970, 937814.0])
paramsDD_FWHM = np.array([82.542, 1.7013e-3, 0.16888, 0.49, 7.9460e-4, 8.4619e-3, 8.3241e-4])
csv_theory = reaction_rate(Tion_theory,paramsDD_csv)
FWHM_theory = FWHM(Tion_theory, paramsDD_FWHM)

plt.figure(num=1)
plt.xlabel('En(MeV)')
plt.ylabel('Probability Density')
plt.title('Normalized Energy Distribution')
plt.legend()
plt.show()

# print(np.size(pd_ux2))
# print(pd_idx_list)
# print(pd_dead2[0])
# print(pd_w2.sum())
# print(tot_probability_)
# print(csv)
#plt.figure(num=1)        
# plt.plot(time/1e-15, Tex*0.511e6, label='Tex')
# plt.plot(time/1e-15, Tey*0.511e6, label='Tey')
# plt.plot(time/1e-15, Tez*0.511e6, label='Tez')
# plt.plot(time/1e-15, Tmean2*0.511e6, label='T2')
# plt.xlabel('Time (fs)')
# plt.ylabel('Temperature (eV)')
# plt.title('Temperature vs Time')
# max_length = max(len(Tion_list), len(csv), len(FWHM_sim))

# Tion_list = np.pad(Tion_list, (0, max_length - len(Tion_list)), constant_values=np.nan)
# csv       = np.pad(csv,       (0, max_length - len(csv)),       constant_values=np.nan)
# FWHM_sim  = np.pad(FWHM_sim,  (0, max_length - len(FWHM_sim)),  constant_values=np.nan)

# data = pd.DataFrame({
#     'Tion (keV)': Tion_list,
#     'csv (cm^3/s)': csv*1.e6,
#     'FWHM (keV)': FWHM_sim,
# })
# data.to_csv(r"D:\Desktop\test\simulation.csv", index=False)
# print("成功导出")

# plt.figure(num=1)
# plt.plot(Tion_list, csv*1.e6, 'ro', label='csv')
# plt.plot(Tion_theory, csv_theory, 'b-', label='csv theory')
# plt.xlabel('Tion (keV)')
# plt.ylabel('reaction rate (cm^3/s)')
# plt.legend()
# plt.show()

# plt.figure(num=2)
# plt.plot(Tion_list, FWHM_sim, 'ro', label='FWHM')
# plt.plot(Tion_theory, FWHM_theory, 'b-', label='FWHM theory')
# plt.xlabel('Tion (keV)')
# plt.ylabel('FWHM (keV)')
# plt.legend()
# plt.show()

# plt.figure(num=2)
# plt.plot(ekin_test,probtest, 'o')
# plt.show()

# plt.figure(num=3)
# plt.hist(Kn, bins=100,weights=pd_w2, color='blue', edgecolor='black')
# plt.show()


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