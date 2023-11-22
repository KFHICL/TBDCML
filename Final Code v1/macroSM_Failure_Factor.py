# Script to generate the magnitude of stress for failure for a given global unit loading vector! Script ran at each seed point in FE framework.
from microSM_stoch import *
from mesoSM_ITSC_SurfGen import *
from macroSM_Intersection_Finder import *
import random as rnd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Time execution of code
import time
start_time = time.time()

###################################################################################
# Geometry Input Parameters ------------------------------------------------------%
t_plate = 1 # mm - Thickness of the TBDC component at the point in consideration
t_ply = 0.125 # mm - Thickness of a ply, which is the same as the tow thickness for a plane ply
l_t,t_t,w_t = 50,t_ply,8 # mm - Tow geometry (length, thickness and width)
theta_EL_1, theta_EL_2 = (-np.pi/2), np.pi/2 # Range of angles at which plies can be oriented
n_overlaps = 4 # The number of overlaps stochastically modelled at the micro-scale
# Tow Property Input Parameters --------------------------------------------------%
E_1_UD,X_T_UD,V_f_UD,V_f = 116000,1132,0.513,0.593 # Measured Longitudinal
# Modulus and Longitudinal Strength for measured vibre volume fraction, 
# scaled to real fibre volume fraction. (All in MPa except volume fractions)
G_II_c = 0.80 # Fracture toughness in kJ/m^2
S = 78 # Shear Strength in MPa
E_2 = 9000 # Transverse Modulus in MPa
G_12 = 5600 # Shear Modulus in MPa
poissons_12 = 0.34 # Poissons ratio
Y_t = 73 # Transverse tensile strength in MPa
Y_c = 200 # Transverse compressive strength in MPa
alpha= 53.0* np.pi/180 # Angle of fracture surfaces - determined experimentally and hence is 
# specific to composite in consideration (for thick plies)
# Number Input Parameters --------------------------------------------------------%
N_e = 200
N_tau12,N_sigma2 = 50,50 ### IMPORTANT - control speed of script -> initially set to 100 each!
# Loading Vector Input Parameters ------------------------------------------------%
sigma_g_unnormalised = [Alpha,Beta,Gamma] = [1,0,0] # global loading vector in X,Y,S direction (any magnitude)
sigma_g = sigma_g_unnormalised/np.linalg.norm(sigma_g_unnormalised) 
# unit global loading vector in X,Y,S direction (any magnitude)
###################################################################################

# Generated Properties -----------------------------------------------------------%
X_T = X_T_UD*(V_f/V_f_UD) # Longitudinal Tensile Strength in MPa
E_1 = E_1_UD*(V_f/V_f_UD) # Young's Modulus in MPa

# Decomposing Global Loading Vector to Local Loading Vectors ---------------------%
# Using classical laminate theory
Q_0 = np.zeros((3,3)) # Stiffnes matrix for theta = 0
Q_0[0][0] = (E_1**2)/(E_1-(poissons_12*E_2))
Q_0[0][1] = Q_0[1][0] = (poissons_12*E_1*E_2)/(E_1-((poissons_12**2)*E_2))
Q_0[1][1] = (E_1*E_2)/(E_1-((poissons_12**2)*E_2))
Q_0[2][2] = G_12

def Q_0_to_Q_i(theta,Q_0): # Rotating Q_0 for each ply (defining terms instead of matrix multiplication)
    Q_i = np.zeros((3,3))
    Q_i[0][0] = Q_0[0][0]*(np.cos(theta)**4) + 2*(Q_0[0][1]+2*Q_0[2][2])*(np.cos(theta)**2)*(np.sin(theta)**2) + Q_0[1][1]*(np.sin(theta)**4)
    Q_i[0][1] = Q_i[1][0] = Q_0[0][1]*((np.cos(theta)**4)+(np.sin(theta)**4)) + (Q_0[0][0]+Q_0[1][1]-4*Q_0[2][2])*(np.cos(theta)**2)*(np.sin(theta)**2)
    Q_i[0][2] = Q_i[2][0] = (Q_0[0][0]-Q_0[0][1]-2*Q_0[2][2])*(np.cos(theta)**3)*np.sin(theta) - (Q_0[1][1]-Q_0[0][1]-2*Q_0[2][2])*np.cos(theta)*(np.sin(theta)**3)
    Q_i[1][1] = Q_0[0][0]*(np.sin(theta)**4) + 2*(Q_0[0][1]+2*Q_0[2][2])*(np.cos(theta)**2)*(np.sin(theta)**2) + Q_0[1][1]*(np.cos(theta)**4)
    Q_i[1][2] = Q_i[2][1] = (Q_0[0][0]-Q_0[0][1]-2*Q_0[2][2])*np.cos(theta)*(np.sin(theta)**3) - (Q_0[1][1]-Q_0[0][1]-2*Q_0[2][2])*(np.cos(theta)**3)*np.sin(theta)
    Q_i[2][2] = (Q_0[0][0]+Q_0[1][1]-2*Q_0[0][1]-2*Q_0[2][2])*(np.cos(theta)**2)*(np.sin(theta)**2) + Q_0[2][2]*((np.cos(theta)**4)+(np.sin(theta)**4))
    return Q_i

def Transform_Matrix(theta): # Define transformation matrix
    T = np.zeros((3,3))
    T[0][0] = T[1][1] = np.cos(theta)**2
    T[0][1] = T[1][0] = np.sin(theta)**2
    T[0][2] = 2*np.sin(theta)*np.cos(theta)
    T[1][2] = -T[0][2]
    T[2][0] = -np.sin(theta)*np.cos(theta)
    T[2][1] = -T[2][0]
    T[2][2] = ((np.cos(theta)**2)-(np.sin(theta)**2))
    return T

def macroSM(N_tau12,S,N_sigma2,n_overlaps,l_t,w_t,t_t,E_1,X_T,G_II_c,N_e,):
    t_char = (w_t*t_t)/(2*(w_t+t_t)) # Charatceristic Thickness
    N_t = int(t_plate/t_ply) # Number of plys in laminate
    S_is=np.sqrt(2)*S # in-situ shear strength
    Y_is=1.12*np.sqrt(2)*Y_t # in-situ transverse strength
    eta=-S/Y_c*np.cos(2*alpha)/np.cos(alpha)**2 # friction coefficient
    theta_t = [rnd.uniform(theta_EL_1,theta_EL_2) for i in range(N_t)] # Angle of each ply in laminate
    Q_i_N = [Q_0_to_Q_i(theta,Q_0) for theta in theta_t] # All Q_i matrices in one array
    Q_g = np.mean(Q_i_N,axis=0) # Q_g is the weighted average of all Q_i matrices depending on ply thickness (equal here)
    overlap_lenths = list(np.zeros(N_t))
    overlap_thicknesses = list(np.zeros(N_t))
    for i in range(N_t): # generating N_t sets of n_overlap random overlap lengths and thicknesses
        overlap_lenths[i] = [rnd.uniform(0,l_t/2) for j in range(n_overlaps)]
        overlap_thicknesses[i] = [rnd.uniform(0,2*t_char) for j in range(n_overlaps)]
    overlap_lenths = np.array(overlap_lenths)
    overlap_thicknesses = np.array(overlap_thicknesses)

    sigma_l = list(np.zeros(N_t)) # Local loading vectors for each ply at theta
    for i in range(N_t):
        sigma_l[i] = [np.linalg.multi_dot([Q_i_N[i],Transform_Matrix(theta_t[i]),np.linalg.inv(Q_g),np.transpose(sigma_g)])] # generating alpha, beta and gamma for each ply
    sigma_l = np.array(sigma_l) ### GIVES LOCAL LOADING VECTORS TO USE TO FIND FAILURE OF EACH PLY (INTERSECTION WITH EACH FAILURE SURFACE)

    # Calculating Local Failure Factor -----------------------------------------------%
    ### Given X_t[i] and a local loading vector simga_i[i], calculate local failure factor m from ITSC
    ### Need to generate ITSC and find intersection with each local loading vector direction
    local_failure_factors = []
    for index in range(0,N_t):
        # Generate ITSC_Surface for each ply within the laminate - different set of random overlap lengths and thicknesses for eacch
        ITSC_Surf, tau12_input, sigma2_s = ITSC_Generation(N_tau12,S_is,Y_is,eta,N_sigma2,n_overlaps,l_t,E_1,X_T,G_II_c,N_e,overlap_lenths[index],overlap_thicknesses[index])

        # Extract x and y values from ITSC_Surface
        x_values = [curve[0] for curve in ITSC_Surf]
        y_values = [curve[1] for curve in ITSC_Surf]

        x_val,y_val,z_val,local_ply_failure_factor = Intersection_Finder(sigma_l[index],x_values,y_values,sigma2_s)
        local_failure_factors += [round(local_ply_failure_factor,2)] 
        ### Magnitude of the applied loading given the applied loading direction to cause failure of each ply within the laminate

    return local_failure_factors

failure_factors = []
count = 0
while count < 50:
    failure_factors += [macroSM(N_tau12,S,N_sigma2,n_overlaps,l_t,w_t,t_t,E_1,X_T,G_II_c,N_e)[2]]
    count += 1
failure_factors = np.sort(np.ravel(failure_factors))

#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density
 
#Calculate mean and Standard deviation.
mean = np.mean(failure_factors)
sd = np.std(failure_factors)
print(mean,sd)
 
#Apply function to the data.
pdf = normal_dist(failure_factors,mean,sd)
 
#Plotting the Results
plt.plot(failure_factors, norm.pdf(failure_factors, mean, sd) , color = 'red')
plt.xlabel('Predicted Composite Failure Stress via FPFM [MPa]')
plt.ylabel('Probability')
plt.grid()
plt.show()
print("--- %s seconds ---" % (time.time() - start_time)) # Print time of execution