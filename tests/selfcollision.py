import numpy as np
from libpic.collision.cpu import  self_collision_parallel_2d
from scipy.stats import gamma
from scipy.constants import pi, m_e, e, c, m_u, epsilon_0
import matplotlib.pyplot as plt
#t_step = np.array([0.1e-17, 0.2e-17, 0.5e-17, 1e-17, 2e-17, 5e-17, 1e-16])
t_step = np.array([ 0.1e-17, 0.2e-17, 0.5e-17, 1e-17, 5e-17])
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.figure(num=1)

for i, step in enumerate(t_step):
    nx = ny = 20
    dx = 0.1e-6
    dy = 0.1e-6
    dt = step

    m = m_e
    q = -e

    ppc = 4000
    # npart_in_cell = np.random.randint(100, 150, nx*ny)
    npart_in_cell = np.full(nx*ny, ppc)
    cell_bound_max = np.cumsum(npart_in_cell)
    cell_bound_min = cell_bound_max - npart_in_cell
    nbuf = npart_in_cell.sum()

    cell_bound_min = np.reshape(cell_bound_min, (nx, ny))
    cell_bound_max = np.reshape(cell_bound_max, (nx, ny))

    dead = np.random.uniform(size=nbuf) < 0.

    Tex0 = 0.000011
    Tey0 = Tez0 = 0.00001
    E = gamma(a=3/2, scale=Tex0).rvs(nbuf)
    E1 = gamma(a=3/2, scale=Tey0).rvs(nbuf)        
    phi = np.arccos(np.random.uniform(-1, 1, nbuf))
    theta = np.random.uniform(0, 2*pi, nbuf)
    beta = np.sqrt(2*E)
    beta1 = np.sqrt(2*E1)
    betax = beta * np.cos(theta) * np.sin(phi)
    betay = beta1 * np.sin(theta) * np.sin(phi)
    betaz = beta1 * np.cos(phi)
    inv_gamma2 = np.sqrt(1 - (betax**2 + betay**2 + betaz**2))
    ux = betax / inv_gamma2
    uy = betay / inv_gamma2 #* np.sqrt(Tey0/Tex0)
    uz = betaz / inv_gamma2 #* np.sqrt(Tez0/Tex0)


    ne =  1.116e28
    w = np.random.uniform(1, 1, nbuf) * ne * dx * dy / ppc
    #w = np.ones(nbuf) * ne * dx * dy / ppc

    w[dead] = 0

    coulomb_log = 2.0
    random_gen = np.random.default_rng()


    test_time = 2e-15
    nsteps = int(test_time / dt)
    Tex = np.zeros(nsteps+1)
    Tey = np.zeros(nsteps+1)
    Tez = np.zeros(nsteps+1)
    Tmean = np.zeros(nsteps+1)
    # m_macro = m * w
    # q_macro = q * w
    # betax = ux * inv_gamma2
    # betay = uy * inv_gamma2
    # betaz = uz * inv_gamma2
    # Tex[0] = np.mean(betax**2)
    # Tey[0] = np.mean(betay**2)
    # Tez[0] = np.mean(betaz**2)
    # Tmean[0] = (Tex[0] + Tey[0] + Tez[0]) / 3

    ux0 = (ux * w).sum() / w.sum()
    uy0 = (uy * w).sum() / w.sum()
    uz0 = (uz * w).sum() / w.sum()
    Tex_ = ((ux-ux0)**2) # mc2
    Tey_ = ((uy-uy0)**2) # mc2
    Tez_ = ((uz-uz0)**2) # mc2
    Tex[0] = ((Tex_ * w).sum() / w.sum())
    Tey[0] = ((Tey_ * w).sum() / w.sum())
    Tez[0] = ((Tez_ * w).sum() / w.sum())
    Tmean[0] = (Tex[0] + Tey[0] + Tez[0]) / 3

    time = np.arange(nsteps+1) * dt
    for _ in range(nsteps):
        self_collision_parallel_2d(
            cell_bound_min, cell_bound_max, 
            nx, ny, dx, dy, dt, 
            ux, uy, uz, inv_gamma2, w, dead, 
            m, q, coulomb_log, random_gen
        )
        # betax = ux * inv_gamma2
        # betay = uy * inv_gamma2
        # betaz = uz * inv_gamma2
        # Tex[_+1] = np.mean(betax**2)
        # Tey[_+1] = np.mean(betay**2)
        # Tez[_+1] = np.mean(betaz**2)
        # Tmean[_+1] = (Tex[_+1] + Tey[_+1] + Tez[_+1]) / 3

        ux0 = (ux * w).sum() / w.sum()
        uy0 = (uy * w).sum() / w.sum()
        uz0 = (uz * w).sum() / w.sum()
        Tex_ = ((ux-ux0)**2) # mc2
        Tey_ = ((uy-uy0)**2) # mc2
        Tez_ = ((uz-uz0)**2) # mc2
        Tex[_+1] = ((Tex_ * w).sum() / w.sum())
        Tey[_+1] = ((Tey_ * w).sum() / w.sum())
        Tez[_+1] = ((Tez_ * w).sum() / w.sum())
        Tmean[_+1] = (Tex[_+1] + Tey[_+1] + Tez[_+1]) / 3

    Tpar  = Tex[0]
    Tperp = (Tey[0]+Tez[0])/2
    dt_theory = dt/100
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
        # nu0 = coeff * density_electron * coulomb_log /Tpar**1.5 * A**-2 *(
        #     -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )
        
        nu0 = 2.*np.sqrt(pi)*(e**2/(4*pi*epsilon_0))**2*ne*coulomb_log/(np.sqrt(m_e)*(Tpar*m_e*c**2)**1.5) * A**-2 *(
            -3. + (A+3.) * np.arctanh((-A)**0.5)/(-A)**0.5 )# * 1e-6 # 1e-6 to convert to seconds

        #print A, Tpar, Tperp, nu0
        Tpar  -= 2.*nu0*(Tpar-Tperp)* dt_theory
        Tperp +=    nu0*(Tpar-Tperp)* dt_theory
    plt.figure(num=1)
    plt.plot(time/1e-15, Tex*0.511e6, label=f'dt={step}', color=colors[i % len(colors)])
    plt.plot(time/1e-15, (Tey+Tez)/2*0.511e6, color=colors[i % len(colors)])
    

#print(w[1],w[2],w[3])

# plt.plot(t_theory/1e-15, Tpar_theory*0.511e6, label='Tpar_theory', color='black')
# plt.plot(t_theory/1e-15, Tperp_theory*0.511e6, label='Tperp_theory', color='black')
# plt.plot(time/1e-15, Tex*0.511e6, label='Tpar', color='red')
# plt.plot(time/1e-15, (Tey+Tez)/2*0.511e6, label='Tperp', color='blue')
#plt.plot(time/1e-15, Tmean*0.511e6, label='Tmean')
plt.figure(num=1)
plt.xlabel('Time (fs)')
plt.ylabel('Temperature (eV)')
plt.title('Temperature vs Time')
plt.legend()
plt.show()

# plt.figure(num=2)
# plt.plot(time/1e-15, Tmean*0.511e6, label='Tmean')