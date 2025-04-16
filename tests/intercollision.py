
import numpy as np
from libpic.collision.cpu import  inter_collision_parallel_2d
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u, epsilon_0, k
import matplotlib.pyplot as plt

nx = ny = 20
dx = 0.1e-6
dy = 0.1e-6
dt = 0.1e-16

m1 = m_e
Z2 = 1
q1 = -e*Z2
m2 = 2*m_e
Z1 = 1
q2 = -e*Z1

ppc1 = 4000
# npart_in_cell = np.random.randint(100, 150, nx*ny)
npart_in_cell1 = np.full(nx*ny, ppc1)
cell_bound_max1 = np.cumsum(npart_in_cell1)
cell_bound_min1 = cell_bound_max1 - npart_in_cell1
nbuf1 = npart_in_cell1.sum()
cell_bound_min1 = np.reshape(cell_bound_min1, (nx, ny))
cell_bound_max1 = np.reshape(cell_bound_max1, (nx, ny))

ppc2 = 4000
npart_in_cell2 = np.full(nx*ny, ppc2)
cell_bound_max2 = np.cumsum(npart_in_cell2)
cell_bound_min2 = cell_bound_max2 - npart_in_cell2
nbuf2 = npart_in_cell2.sum()
cell_bound_min2 = np.reshape(cell_bound_min2, (nx, ny))
cell_bound_max2 = np.reshape(cell_bound_max2, (nx, ny))

dead1 = np.random.uniform(size=nbuf1) < 0.
dead2 = np.random.uniform(size=nbuf2) < 0.

T1 = 20/511e3
T2 = 10/511e3
E1 = gamma(a=3/2, scale=T1).rvs(nbuf1)
E2 = gamma(a=3/2, scale=T2).rvs(nbuf2)

phi1 = np.arccos(np.random.uniform(-1, 1, nbuf1))
theta1 = np.random.uniform(0, 2*pi, nbuf1)
phi2 = np.arccos(np.random.uniform(-1, 1, nbuf2))
theta2 = np.random.uniform(0, 2*pi, nbuf2)

beta1 = np.sqrt(2*E1*m_e/m1) #np.sqrt(1.-1./(E1*m_e/m1+1)**2)
beta2 = np.sqrt(2*E2*m_e/m2) #np.sqrt(1.-1./(E2*m_e/m2+1)**2)
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

ne1 = 1.116e28
ne2 = 1.116e28
w1 = np.random.uniform(1, 1, nbuf1) * ne1 * dx * dy / ppc1
w2 = np.random.uniform(1, 1, nbuf2) * ne2 * dx * dy / ppc2
w1[dead1] = 0
w2[dead2] = 0

coulomb_log = 2.0
random_gen = np.random.default_rng()

test_time = 15e-15
nsteps = int(test_time / dt)
Tex = np.zeros(nsteps+1)
Tey = np.zeros(nsteps+1)
Tez = np.zeros(nsteps+1)

Tex1 = np.zeros(nsteps+1)
Tey1 = np.zeros(nsteps+1)
Tez1 = np.zeros(nsteps+1)
Tmean1 = np.zeros(nsteps+1)
# betax1 = ux1 * inv_gamma1
# betay1 = uy1 * inv_gamma1
# betaz1 = uz1 * inv_gamma1
# Tex1[0] = np.mean(betax1**2*m1/m_e)
# Tey1[0] = np.mean(betay1**2*m1/m_e)
# Tez1[0] = np.mean(betaz1**2*m1/m_e)
# Tmean1[0] = (Tex1[0] + Tey1[0] + Tez1[0]) / 3

ux10 = (ux1 * w1).sum() / w1.sum()
uy10 = (uy1 * w1).sum() / w1.sum()
uz10 = (uz1 * w1).sum() / w1.sum()
Tex1_ = ((ux1-ux10)**2)*m1/m_e # mc2
Tey1_ = ((uy1-uy10)**2)*m1/m_e # mc2
Tez1_ = ((uz1-uz10)**2)*m1/m_e # mc2
Tex1[0] = ((Tex1_ * w1).sum() / w1.sum())
Tey1[0] = ((Tey1_ * w1).sum() / w1.sum())
Tez1[0] = ((Tez1_ * w1).sum() / w1.sum())
Tmean1[0] = (Tex1[0] + Tey1[0] + Tez1[0]) / 3
#Tmean1[0] = m1/m_e * ((1./inv_gamma1-1)*w1).sum() / w1.sum()

Tex2 = np.zeros(nsteps+1)
Tey2 = np.zeros(nsteps+1)
Tez2 = np.zeros(nsteps+1)
Tmean2 = np.zeros(nsteps+1)
# betax2 = ux2 * inv_gamma2
# betay2 = uy2 * inv_gamma2
# betaz2 = uz2 * inv_gamma2
# Tex2[0] = np.mean(betax2**2*m2/m_e)
# Tey2[0] = np.mean(betay2**2*m2/m_e)
# Tez2[0] = np.mean(betaz2**2*m2/m_e)
# Tmean2[0] = (Tex2[0] + Tey2[0] + Tez2[0]) / 3

ux20 = (ux2 * w2).sum() / w2.sum()
uy20 = (uy2 * w2).sum() / w2.sum()
uz20 = (uz2 * w2).sum() / w2.sum()
Tex2_ = ((ux2-ux20)**2)*m2/m_e
Tey2_ = ((uy2-uy20)**2)*m2/m_e
Tez2_ = ((uz2-uz20)**2)*m2/m_e
Tex2[0] = ((Tex2_ * w2).sum() / w2.sum())
Tey2[0] = ((Tey2_ * w2).sum() / w2.sum())
Tez2[0] = ((Tez2_ * w2).sum() / w2.sum())
Tmean2[0] = (Tex2[0] + Tey2[0] + Tez2[0]) / 3
#Tmean2[0] = m2/m_e * ((1./inv_gamma2-1)*w2).sum() / w2.sum()

Tex[0] = ((Tex1_ * w1).sum() + (Tex2_ * w2).sum()) / (w1.sum()+w2.sum())
Tey[0] = ((Tey1_ * w1).sum() + (Tey2_ * w2).sum()) / (w1.sum()+w2.sum())
Tez[0] = ((Tez1_ * w1).sum() + (Tez2_ * w2).sum()) / (w1.sum()+w2.sum())

time = np.arange(nsteps+1) * dt
for _ in range(nsteps):
    inter_collision_parallel_2d(
        cell_bound_min1, cell_bound_max1, cell_bound_min2, cell_bound_max2,
        nx, ny, dx, dy, dt,
        ux1, uy1, uz1, inv_gamma1, w1, dead1,
        ux2, uy2, uz2, inv_gamma2, w2, dead2,
        m1, q1, m2, q2,
        coulomb_log, random_gen
    )
    # betax1 = ux1 * inv_gamma1
    # betay1 = uy1 * inv_gamma1
    # betaz1 = uz1 * inv_gamma1
    # Tex1[_+1] = np.mean(betax1**2*m1/m_e)
    # Tey1[_+1] = np.mean(betay1**2*m1/m_e)
    # Tez1[_+1] = np.mean(betaz1**2*m1/m_e)
    # Tmean1[_+1] = (Tex1[_+1] + Tey1[_+1] + Tez1[_+1]) / 3

    # betax2 = ux2 * inv_gamma2
    # betay2 = uy2 * inv_gamma2
    # betaz2 = uz2 * inv_gamma2
    # Tex2[_+1] = np.mean(betax2**2*m2/m_e)
    # Tey2[_+1] = np.mean(betay2**2*m2/m_e)
    # Tez2[_+1] = np.mean(betaz2**2*m2/m_e)
    # Tmean2[_+1] = (Tex2[_+1] + Tey2[_+1] + Tez2[_+1]) / 3

    ux10 = (ux1 * w1).sum() / w1.sum()
    uy10 = (uy1 * w1).sum() / w1.sum()
    uz10 = (uz1 * w1).sum() / w1.sum()
    Tex1_ = ((ux1-ux10)**2)*m1/m_e # mc2
    Tey1_ = ((uy1-uy10)**2)*m1/m_e # mc2
    Tez1_ = ((uz1-uz10)**2)*m1/m_e # mc2
    Tex1[_+1] = ((Tex1_ * w1).sum() / w1.sum())
    Tey1[_+1] = ((Tey1_ * w1).sum() / w1.sum())
    Tez1[_+1] = ((Tez1_ * w1).sum() / w1.sum())
    Tmean1[_+1] = (Tex1[_+1] + Tey1[_+1] + Tez1[_+1]) / 3

    ux20 = (ux2 * w2).sum() / w2.sum()
    uy20 = (uy2 * w2).sum() / w2.sum()
    uz20 = (uz2 * w2).sum() / w2.sum()
    Tex2_ = ((ux2-ux20)**2)*m2/m_e
    Tey2_ = ((uy2-uy20)**2)*m2/m_e
    Tez2_ = ((uz2-uz20)**2)*m2/m_e
    Tex2[_+1] = ((Tex2_ * w2).sum() / w2.sum())
    Tey2[_+1] = ((Tey2_ * w2).sum() / w2.sum())
    Tez2[_+1] = ((Tez2_ * w2).sum() / w2.sum())
    Tmean2[_+1] = (Tex2[_+1] + Tey2[_+1] + Tez2[_+1]) / 3

    Tex[_+1] = ((Tex1_ * w1).sum() + (Tex2_ * w2).sum()) / (w1.sum()+w2.sum())
    Tey[_+1] = ((Tey1_ * w1).sum() + (Tey2_ * w2).sum()) / (w1.sum()+w2.sum())
    Tez[_+1] = ((Tez1_ * w1).sum() + (Tez2_ * w2).sum()) / (w1.sum()+w2.sum())


T1 = Tmean1[0]
T2 = Tmean2[0]
dt_theory = dt/100
t_theory = np.arange(test_time/dt_theory) * dt_theory
T1_theory  = np.zeros_like(t_theory)
T2_theory = np.zeros_like(t_theory)
for it in range(len(t_theory)):
    T1_theory[it] = T1
    T2_theory[it] = T2       
    nu0 = 1./3.*np.sqrt(2./pi) * (e**4*Z1**2*Z2**2*np.sqrt(m1*m2)*ne2*coulomb_log) / (
        4.*pi*epsilon_0**2*(m1*T1*m_e*c**2 + m2*T2*m_e*c**2)**1.5)
    
    dT = nu0 * (T1 - T2) * dt_theory
    T1 -= dT
    T2 += dT
    # T1 -= nu0*(T1-T2)* dt_theory
    # T2 += nu0*(T1-T2)* dt_theory

print(Tex[0], Tey[0], Tez[0])
plt.figure(num=2)
plt.plot(time/1e-15, Tmean1*0.511e6, label='T1')
plt.plot(time/1e-15, Tmean2*0.511e6, label='T2')
plt.plot(t_theory/1e-15, T1_theory*0.511e6, label='T1(T2)theory', color='black')
plt.plot(t_theory/1e-15, T2_theory*0.511e6, color='black')
plt.xlabel('Time (fs)')
plt.ylabel('Temperature (eV)')
plt.title('Temperature vs Time')
plt.legend()
plt.show()

# Tpar  = Tex[0]
# Tperp = (Tey[0]+Tez[0])/2
# dt_theory = dt/100
# t_theory = np.arange(test_time/dt_theory) * dt_theory
# Tpar_theory  = np.zeros_like(t_theory)
# Tperp_theory = np.zeros_like(t_theory)

# re_ = 2.8179403267e-15 # meters
# wavelength =  1e-6 # meters
# coeff = (2*pi/wavelength)**2*re_*c / (2.*np.sqrt(pi))
# density_electron = 10.01027741148843

# for it in range(len(t_theory)):
#     Tpar_theory[it] = Tpar
#     Tperp_theory[it] = Tperp
#     A = Tperp/Tpar - 1.
#     if A>0: 
#         break
#     nu0 = coeff * density_electron * coulomb_log /Tpar**1.5 * A**-2 *(
#         -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )
#     # nu0 = 2.*np.sqrt(pi)*(e**2/(4*pi*epsilon_0))**2*ne*coulomb_log/(np.sqrt(m_e)*(Tpar*m_e*c**2)**1.5) * A**-2 *(
#     #     -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )# * 1e-6 # 1e-6 to convert to seconds

#     #print A, Tpar, Tperp, nu0
#     Tpar  -= 2.*nu0*(Tpar-Tperp)* dt_theory
#     Tperp +=    nu0*(Tpar-Tperp)* dt_theory

# plt.figure(num=2)
# plt.plot(t_theory/1e-15, Tpar_theory*0.511e6, label='Tpar_theory', color='black')
# plt.plot(t_theory/1e-15, Tperp_theory*0.511e6, label='Tperp_theory', color='black')
# plt.plot(time/1e-15, Tex*0.511e6, label='Tpar', color='red')
# plt.plot(time/1e-15, (Tey+Tez)/2*0.511e6, label='Tperp', color='blue')
# plt.xlabel('Time (fs)')
# plt.ylabel('Temperature (eV)')
# plt.title('Temperature vs Time')
# plt.legend()
# plt.show()
