import numpy as np
from numba import njit, prange
from scipy.constants import c, pi, epsilon_0, m_e, e
from libpic.collision.cpu import pairing
from math import sqrt
##uint？
# @njit(cache=True)
# def cross_sectionDD(log_ekin):
    # npoints = 50
    # npointsm1 = npoints-1. 
    # a1 = np.log(511./(2.*2.013553));  # = ln(me*c^2 / Emin / n_nucleons)
    # a2 = 3.669039;  # = (npoints-1) / ln( Emax/Emin )
    # DB_log_crossSection = [
    # -27.307, -23.595, -20.383, -17.607, -15.216, -13.167, -11.418, -9.930, -8.666, -7.593,
    # -6.682, -5.911, -5.260, -4.710, -4.248, -3.858, -3.530, -3.252, -3.015, -2.814, -2.645,
    # -2.507, -2.398, -2.321, -2.273, -2.252, -2.255, -2.276, -2.310, -2.352, -2.402, -2.464,
    # -2.547, -2.664, -2.831, -3.064, -3.377, -3.773, -4.249, -4.794, -5.390, -6.021, -6.677,
    # -7.357, -8.072, -8.835, -9.648, -10.498, -11.387, -12.459]
    # x = a2*( a1 + log_ekin )
    # if x<0:
    #      cs = 0.0
    # elif x < npointsm1:
    #      i = int( x )
    #      a = x - i
    #      cs = np.exp(( DB_log_crossSection[i+1]-DB_log_crossSection[i] )*a + DB_log_crossSection[i])
    # else:
    #      a = x - npointsm1
    #      cs = np.exp(( DB_log_crossSection[npoints-1]-DB_log_crossSection[npoints-2] )*a + DB_log_crossSection[npoints-1])
    
    # return cs
@njit(cache=True)
def cross_sectionDD(Ekin_com):  #Ekin_com-keV,cs-mbarn
    S = (5.3701e4 + Ekin_com*(3.3027e2 + Ekin_com*(-1.2706e-1 + Ekin_com*(2.9327e-5 + Ekin_com*-2.515e-9)))) / 1.
        
    cs = S/(Ekin_com*np.exp(31.3970/np.sqrt(Ekin_com)))
    return cs


def reaction_rate(Tion, params): #Tion-keV, csv-cm^3/s
    C1, C2, C3, C4, C5, C6, C7, BG, mrc2 = params[:9]
    theta = Tion/(1.-(Tion*(C2+Tion*(C4+Tion*C6)))/(1.+Tion*(C3+Tion*(C5+Tion*C7))))
    eps = (BG**2/4./theta)**(1./3.)
    csv = C1*theta*np.sqrt(eps/(mrc2*Tion**3))*np.exp(-3.*eps)
    return csv

def FWHM(Tion, params): #Tion-keV, FWHM-keV
    omega0, a1, a2, a3, a4, a5, a6 = params[:7]
    # if Tion < 30.0:
    #     delta = (a1/(1. + a2*Tion**a3)) * Tion**(2./3.) + a4*Tion 
    # else:
    #     delta = a5 + a6*Tion

    delta = np.where(
        Tion < 30.0,
        (a1 / (1. + a2 * Tion**a3)) * Tion**(2./3.) + a4 * Tion,  # 条件为 True 时的值
        a5 + a6 * Tion  # 条件为 False 时的值
    )
    result = omega0*(1.+delta)*np.sqrt(Tion)
    return result

@njit(cache=True)
def DDfusion_cell_2d(
    ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
    m1, q1, m2, q2,
    pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
    pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
    pd_m1, pd_q1, pd_m2, pd_q2,
    lnLambda,
    dx, dy, dt,
    random_gen, rate_multiplier_, tot_probability_, pd_idx_list #, iproduct, probtest, ekin_test
):
    nbuf1 = ip_end1 - ip_start1
    npart1 = nbuf1 - dead1[ip_start1:ip_end1].sum()

    nbuf2 = ip_end2 - ip_start2
    npart2 = nbuf2 - dead2[ip_start2:ip_end2].sum()
    if npart1 == 0 or npart2 == 0: 
        return

    npairs = max(npart1, npart2)
    dt_corr = npairs

    # loop pairs
    for ipair, ip1, ip2, w_corr in pairing(
        dead1, ip_start1, ip_end1, 
        dead2, ip_start2, ip_end2, random_gen
    ):
        w1_ = w1[ip1] * w_corr
        w2_ = w2[ip2] * w_corr
        w_max = max(w1_, w2_)
        w_min = min(w1_, w2_)
        


        gamma1 = 1/inv_gamma1[ip1]
        gamma2 = 1/inv_gamma2[ip2]

        p1x = ux1[ip1]*m1*c
        p1y = uy1[ip1]*m1*c
        p1z = uz1[ip1]*m1*c

        p2x = ux2[ip2]*m2*c
        p2y = uy2[ip2]*m2*c
        p2z = uz2[ip2]*m2*c

        v1x = p1x / gamma1 / m1
        v1y = p1y / gamma1 / m1
        v1z = p1z / gamma1 / m1

        v2x = p2x / gamma2 / m2
        v2y = p2y / gamma2 / m2
        v2z = p2z / gamma2 / m2

        vx_com = (p1x + p2x) / (gamma1*m1 + gamma2*m2)
        vy_com = (p1y + p2y) / (gamma1*m1 + gamma2*m2)
        vz_com = (p1z + p2z) / (gamma1*m1 + gamma2*m2)
        gamma_com = 1.0 / sqrt(1 - (vx_com**2 + vy_com**2 + vz_com**2)/c**2 )
        v_com_square = vx_com**2 + vy_com**2 + vz_com**2
        m_reduced = m1*m2/(m1*gamma1+m2*gamma2)

        # _, p1x_com, p1y_com, p1z_com = lorentz_boost(gamma1*m1*c, p1x, p1y, p1z, vx_com, vy_com, vz_com, gamma_com)
        p1x_com = p1x + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vx_com
        p1y_com = p1y + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vy_com
        p1z_com = p1z + ((gamma_com-1)/v_com_square * (v1x*vx_com + v1y*vy_com + v1z*vz_com) - gamma_com) * m1*gamma1*vz_com

        # p2x_com = p2x + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vx_com
        # p2y_com = p2y + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vy_com
        # p2z_com = p2z + ((gamma_com-1)/v_com_square * (v2x*vx_com + v2y*vy_com + v2z*vz_com) - gamma_com) * m2*gamma2*vz_com

        p1_com = np.sqrt(p1x_com**2 + p1y_com**2 + p1z_com**2)

        gamma1_com = (1 - (vx_com*v1x + vy_com*v1y + vz_com*v1z) / c**2) * gamma_com*gamma1
        gamma2_com = (1 - (vx_com*v2x + vy_com*v2y + vz_com*v2z) / c**2) * gamma_com*gamma2

        p_perp = sqrt( p1x_com**2 + p1y_com**2 )
        
        #Interpolate the total cross-section at some value of ekin = m1(g1-1) + m2(g2-1)
        ekin = (m1*(gamma1_com-1.) + m2*(gamma2_com-1.))*c**2   #J
        log_ekin = np.log( ekin/m_e/c**2 ) #J

        cs = cross_sectionDD(ekin/e/1000.)*1.0e-31 #mbarn to m2
        vrel_corr = (m1*gamma1+m2*gamma2)*p1_com/(m1*gamma1*m2*gamma2*gamma_com)
        prob = (npart1+npart2-1)*rate_multiplier_*w_max*cs*vrel_corr*dt/(dx*dy)

        # cs = cross_sectionDD( log_ekin )
        # reference_angular_frequency_SI = 1e-6
        # coeff2_ = 2.817940327e-15 * reference_angular_frequency_SI / c
        # prob = coeff2_ * vrel_corr/c * dt*w_max*reference_angular_frequency_SI * cs * rate_multiplier_
        # prob = (npart1+npart2-1)*rate_multiplier_*w_max*cs*4.*pi*2.817940327e-15**2*vrel_corr*dt/(dx*dy)
 
        # probtest[ipair] = prob
        # ekin_test[ipair] = ekin
        tot_probability_[0] += prob
        #print(tot_probability_)
        #npairs_tot_ ++
        
        if random_gen.uniform() > (1.0-prob): #np.exp( -prob ):
            pd_idx = pd_idx_list[0]
            pd_idx_list[0] = pd_idx + 1
            #pd_idx = np.where(pd_dead1 == True)[0][0]
            w_p = w_min/rate_multiplier_
            w1[ip1] = w1[ip1]-w_p
            w2[ip2] = w2[ip2]-w_p
            pd_w1[pd_idx] = (w_p)
            pd_w2[pd_idx] = (w_p)
            pd_dead1[pd_idx] = (False)
            pd_dead2[pd_idx] = (False)
            #print(pd_idx)
            U = random_gen.uniform(-1, 1)
            Uabs = abs(U)
            up = U>0.0
            lnE = np.log(511./0.7/(2.*2.013553)) + log_ekin
            alpha = 1. if lnE < 0 else np.exp(-0.024 * lnE * lnE)
            one_m_cosX = alpha*Uabs / sqrt( (1.-Uabs) + alpha*alpha*Uabs )
            cosX =  1. - one_m_cosX 
            sinX =  sqrt( one_m_cosX * (1.+cosX) ) 
            phi = np.random.uniform(0, 2*pi)
            cosPhi = np.cos( phi ) 
            sinPhi = np.sin( phi )

            # Calculate the resulting momenta from energy / momentum conservation
            Q = 6.397*m_e*c**2 # Qvalue 单位
            Erest_n = 1838.7*m_e*c**2
            Erest_He = 5497.9*m_e*c**2
            m_He = 5497.9*m_e
            m_n = 1838.7*m_e
            p_COM = sqrt( (ekin+Q) * (ekin+Q+2.*Erest_n) * (ekin+Q+2.*Erest_He) * (ekin+Q+2.*Erest_n+2.*Erest_He) ) / ( 2.* ( ekin+Q+Erest_n+Erest_He ) )/c
            if not up:  
                p_COM = -p_COM
            # new_p_com_He = p_COM / m_He  ##约化动量
            # new_p_COM_n = p_COM / m_n
                
            # # Set particle properties
            # products.q.resize( product_particles_.size() );
            # products.particles.resize( product_particles_.size() );
            # products.new_p_COM.resize( product_particles_.size() );
            # # helium3
            # products.q[index_He_] = (short) tot_charge;
            # products.particles[index_He_] = product_particles_[index_He_];
            # products.new_p_COM[index_He_] = p_COM / m_He; 
            # # neutron
            # if index_n_ <  product_particles_.size():
            #     products.q[index_n_] = (short) 0.
            #     products.particles[index_n_] = product_particles_[index_n_];
            #     products.new_p_COM[index_n_] = - p_COM / m_n
            
            if p_perp > 1.e-10*p1_com : 
                inv_p_perp = 1./p_perp
                newpx_COM = ( p1x_com * p1z_com * cosPhi - p1y_com * p1_com * sinPhi ) * inv_p_perp
                newpy_COM = ( p1y_com * p1z_com * cosPhi + p1x_com * p1_com * sinPhi ) * inv_p_perp
                newpz_COM = -p_perp * cosPhi
            else:
                newpx_COM = p1_com * cosPhi
                newpy_COM = p1_com * sinPhi
                newpz_COM = 0.
            
            # Calculate the deflection in the COM frame
            newpx_COM = newpx_COM * sinX + p1x_com * cosX
            newpy_COM = newpy_COM * sinX + p1y_com * cosX
            newpz_COM = newpz_COM * sinX + p1z_com * cosX
            # Go back to the lab frame and store the results in the particle array
            vcp = vx_com * newpx_COM + vy_com * newpy_COM + vz_com * newpz_COM
            momentum_ratio = p_COM / p1_com
            term6 = momentum_ratio*((gamma_com-1)/v_com_square)*vcp + m_He*sqrt((p_COM*2)/(m_He**2*c**2) + 1. ) * gamma_com
            newpxHe = momentum_ratio * newpx_COM + vx_com * term6
            newpyHe = momentum_ratio * newpy_COM + vy_com * term6
            newpzHe = momentum_ratio * newpz_COM + vz_com * term6

            pd_ux1[pd_idx] = (newpxHe / m_He / c )
            pd_uy1[pd_idx] = (newpyHe / m_He / c )
            pd_uz1[pd_idx] = (newpzHe / m_He / c )
            pd_inv_gamma1[pd_idx] = (1/sqrt((newpxHe**2 + newpyHe**2 + newpzHe**2)/(m_He**2*c**2) + 1) )

            term6 = -momentum_ratio*((gamma_com-1)/v_com_square)*vcp + m_n*sqrt((p_COM**2)/(m_n**2*c**2) + 1. ) * gamma_com
            newpxn = -momentum_ratio * newpx_COM + vx_com * term6
            newpyn = -momentum_ratio * newpy_COM + vy_com * term6
            newpzn = -momentum_ratio * newpz_COM + vz_com * term6
            pd_ux2[pd_idx] = (newpxn / m_n / c )
            pd_uy2[pd_idx] = (newpyn / m_n / c )
            pd_uz2[pd_idx] = (newpzn / m_n / c )
            pd_inv_gamma2[pd_idx] = (1/sqrt((newpxn**2 + newpyn**2 + newpzn**2)/(m_n**2*c**2) + 1) )


        #else:
        s = w_max/dx/dy *dt * (lnLambda * (q1*q2)**2) / (4*pi*epsilon_0**2*c**4 * m1*gamma1 * m2*gamma2) \
                *(gamma_com * p1_com)/(m1*gamma1 + m2*gamma2) * (m1*gamma1_com*m2*gamma2_com/p1_com**2 * c**2 + 1)**2
        s *= dt_corr

        U1 = random_gen.uniform()
        U2 = random_gen.uniform()
        if s < 4:
            alpha = 0.37*s - 0.005*s**2 - 0.0064*s**3
            sin2X2 = alpha * U1 / np.sqrt( (1-U1) + alpha*alpha*U1 )
            cosX = 1. - 2.*sin2X2
            sinX = 2.*np.sqrt( sin2X2 *(1.-sin2X2) )
        else:
            cosX = 2.*U1 - 1.
            sinX = np.sqrt( 1. - cosX*cosX )

        phi = np.random.uniform(0, 2*pi)
        sinXcosPhi = sinX*np.cos( phi )
        sinXsinPhi = sinX*np.sin( phi )
        # make sure p_perp is not too small
        if p_perp > 1.e-10*p1_com:
            p1x_com_new = ( p1x_com * p1z_com * sinXcosPhi - p1y_com * p1_com * sinXsinPhi ) / p_perp + p1x_com * cosX
            p1y_com_new = ( p1y_com * p1z_com * sinXcosPhi + p1x_com * p1_com * sinXsinPhi ) / p_perp + p1y_com * cosX
            p1z_com_new = -p_perp * sinXcosPhi + p1z_com * cosX
        # if p_perp is too small, we use the limit px->0, py=0
        else:
            p1x_com_new = p1_com * sinXcosPhi
            p1y_com_new = p1_com * sinXsinPhi
            p1z_com_new = p1_com * cosX

        vcom_dot_p = vx_com*p1x_com_new + vy_com*p1y_com_new + vz_com*p1z_com_new
        if w2_ / w_max > U2:
            p1x_new = p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            p1y_new = p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            p1z_new = p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * vcom_dot_p + m1*gamma1_com*gamma_com)
            ux1[ip1] = p1x_new / m1 / c
            uy1[ip1] = p1y_new / m1 / c
            uz1[ip1] = p1z_new / m1 / c
            inv_gamma1[ip1] = 1/sqrt(ux1[ip1]**2 + uy1[ip1]**2 + uz1[ip1]**2 + 1)
        if w1_ / w_max > U2:
            p2x_new = -p1x_com_new + vx_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            p2y_new = -p1y_com_new + vy_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            p2z_new = -p1z_com_new + vz_com * ((gamma_com-1)/v_com_square * -vcom_dot_p + m2*gamma2_com*gamma_com)
            ux2[ip2] = p2x_new / m2 / c
            uy2[ip2] = p2y_new / m2 / c
            uz2[ip2] = p2z_new / m2 / c
            inv_gamma2[ip2] = 1/sqrt(ux2[ip2]**2 + uy2[ip2]**2 + uz2[ip2]**2 + 1)

@njit(parallel=True, cache=True)
def DDfusion_parallel_2d(
    cell_bound_min1, cell_bound_max1, 
    cell_bound_min2, cell_bound_max2, 
    nx, ny, dx, dy, dt,
    ux1, uy1, uz1, inv_gamma1, w1, dead1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2,
    m1, q1, m2, q2,
    pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
    pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
    pd_m1, pd_q1, pd_m2, pd_q2,
    lnLambda,
    random_gen, rate_multiplier_, tot_probability_, pd_idx_list #, iproduct, probtest, ekin_test
):
    for icell in prange(nx*ny):
        ix = icell // ny
        iy = icell % ny
        
        ip_start1 = cell_bound_min1[ix,iy]
        ip_end1 = cell_bound_max1[ix,iy]

        ip_start2 = cell_bound_min2[ix,iy]
        ip_end2 = cell_bound_max2[ix,iy]

        DDfusion_cell_2d(
            ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
            ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
            m1, q1, m2, q2,
            pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
            pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
            pd_m1, pd_q1, pd_m2, pd_q2,
            lnLambda,
            dx, dy, dt,
            random_gen,rate_multiplier_, tot_probability_, pd_idx_list #, iproduct, probtest, ekin_test
        )

@njit(cache=True)
def DDfusion_2d(
    cell_bound_min1, cell_bound_max1, 
    cell_bound_min2, cell_bound_max2, 
    nx, ny, dx, dy, dt,
    ux1, uy1, uz1, inv_gamma1, w1, dead1,
    ux2, uy2, uz2, inv_gamma2, w2, dead2,
    m1, q1, m2, q2,
    pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
    pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
    pd_m1, pd_q1, pd_m2, pd_q2,
    lnLambda,
    random_gen, rate_multiplier_, tot_probability_, pd_idx_list#, iproduct, probtest, ekin_test
):
    for ix in range(nx):
        for iy in range(ny):
            
            ip_start1 = cell_bound_min1[ix,iy]
            ip_end1 = cell_bound_max1[ix,iy]

            ip_start2 = cell_bound_min2[ix,iy]
            ip_end2 = cell_bound_max2[ix,iy]

            DDfusion_cell_2d(
                ux1, uy1, uz1, inv_gamma1, w1, dead1, ip_start1, ip_end1,
                ux2, uy2, uz2, inv_gamma2, w2, dead2, ip_start2, ip_end2,
                m1, q1, m2, q2,
                pd_ux1, pd_uy1, pd_uz1, pd_inv_gamma1, pd_w1, pd_dead1,
                pd_ux2, pd_uy2, pd_uz2, pd_inv_gamma2, pd_w2, pd_dead2,
                pd_m1, pd_q1, pd_m2, pd_q2,
                lnLambda,
                dx, dy, dt,
                random_gen,rate_multiplier_, tot_probability_, pd_idx_list#, iproduct, probtest, ekin_test
            )