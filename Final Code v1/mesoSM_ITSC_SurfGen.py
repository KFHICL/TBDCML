from microSM_stoch import *

# %---------------------------------------------------------------%
def meso_ITSC_x_s(N_tau12,n_overlaps,S_is,l_t,E_1,X_T,G_II_c,N_e,l_o_rnd,t_o_rnd): # generate Sigma1 against Tau12 curve
### This curve is applied shear stress v longitudinal tensile strength - defined by calculating the longitduinal shear strength for the values of 
### applied shear stress from the in-situ shear strength to 0, hence longitundal tensile strength will proportionally increase from 0 to the max
### This is done by running the micro-scale model for each value of shear stress, and by plotting the stress-strain curve from the micro-scale
### model at each value of applied shear, you can see how to maximum if the curve changes, and this max value is used to generate the x-s curve!
    applied_shear = [i for i in np.linspace(0,S_is,N_tau12)] # generating range of applied shear stress values from 0 to max shear strength
    ITS_Stress = [microSM(n_overlaps,S_is-i,l_t,E_1,X_T,G_II_c,N_e,l_o_rnd,t_o_rnd) for i in applied_shear[:-1]]
    ### This is generating the stength of a tow/interaction with overlaps defined by the first array in the overlap sets (i.e. out of 8 total)
    ### I.e. generating the x-s curve for one ply/tow/interaction
    ITS_Stress += [0] # Last value, gives 0 tensile strength with the max applied shear stress
    
    return ITS_Stress, applied_shear
# %---------------------------------------------------------------%

# %---------------------------------------------------------------%
# Eqn. 6 in 2019 paper is used to generate this Y-S curve - it is CONSTANT and does not vary with loading!
def local_failure_factor(beta,gamma,Y_is,S_is,eta):
    beta_plus = np.maximum(0,beta);

    if beta_plus <= 10**(-16) or gamma ==1:
        mm = S_is/(gamma+beta*eta)
    else:
        # solve the fourth order polynomial
        a4 = beta**4 * eta**2
        a3 = -2* S_is *beta**3 *eta
        a2 = -Y_is**2*beta**2*eta**2+S_is**2*beta**2+Y_is**2*gamma**2
        a1 = 2*S_is*Y_is**2*beta*eta;
        a0 = -S_is**2*Y_is**2 

        mm4 = np.roots([a4, a3, a2, a1, a0])
        mm_real = np.real(mm4[mm4 == np.real(mm4)])
        m_pos = mm_real[mm_real>0]
        mm = np.min(m_pos)
    
    return mm

def meso_ITSC_y_s(S_is,Y_is,eta,N_sigma2):

    LCs = np.arange(0,np.pi/2 + np.pi/(2*N_sigma2-2),np.pi/(2*N_sigma2-2)) # defines range of 0 to pi/2 with same length as Nsigma2

    ### Populate sigma_y-tau using LaRC05
    ### Defining the tensile strength in the traverse direction against applied shear stress curve - via LaRC05 criterion
    m = np.zeros(N_sigma2) # failure factor
    tau12_y = np.zeros(N_sigma2) # shear stress (v.s. sigma_y)
    sigma2_s = np.zeros(N_sigma2) #sigma_y (v.s. shear)

    for i in range(0,N_sigma2):
        gamma2 = np.sin(LCs[i])
        beta2 = np.cos(LCs[i])
        # create range of beta and gamma that from 0 to 90°

        m[i] = local_failure_factor(beta2,gamma2,Y_is,S_is,eta)
        tau12_y[i]=m[i]*gamma2
        sigma2_s[i]=m[i]*beta2
    tau12_y[tau12_y>S_is]=S_is

    return sigma2_s,tau12_y

# sigma2_s,tau12_y = meso_ITSC_y_s(S,Y_is,eta,N_sigma2)
# Y and S arrays generated for Y-S curve
# %---------------------------------------------------------------%

### S curve is fully defined for each value of Y -> now input each S value into the micro-scale model 
### to generate an X-S curve for each Y value
def ITSC_Generation(N_tau12,S_is,Y_is,eta,N_sigma2,n_overlaps,l_t,E_1,X_T,G_II_c,N_e,l_o_rnd,t_o_rnd):
    sigma2_s,tau12_y = meso_ITSC_y_s(S_is,Y_is,eta,N_sigma2)
    tau12_input = np.flip(tau12_y) # S values to input into meso_ITSC_x_s function - flipped array so S_is is first value
    tau12_input = tau12_input[:-1] # Removing the 0 shear value to avoid dividingby zero error - need to add 0 X-S point at max Y value!
    ITSC_Surface = [] # Need a matrix to hold the X-S curve for EACH Y value, hence shape (N_sigma2)
    for Y_indice in range(0,len(tau12_input)):
        curve = meso_ITSC_x_s(N_tau12,n_overlaps,tau12_input[Y_indice],l_t,E_1,X_T,G_II_c,N_e,l_o_rnd,t_o_rnd) 
        # Inputting S for each Y (from S_is to 0, or 0 to max(Y)) to generate a X-S curve
        ITSC_Surface += [curve]
        ### Works PERFECTLY - gives the X-S curve for INCREASING values of Y
        ### sigma2_s and ITSC_Surf are missing the point/curve related to max(Y),S=0,X=0
    end_point = [0],[0]
    ITSC_Surface += [end_point]

    return ITSC_Surface,tau12_input,np.flip(sigma2_s)