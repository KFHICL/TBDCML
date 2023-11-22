# Script to generate the magnitude of stress for failure for a given global unit loading vector! Script ran at each seed point in FE framework.
from microSM_stoch import *
from mesoSM_ITSC_SurfGen import *
from macroSM_Intersection_Finder import *
import matplotlib.pyplot as plt
import random as rnd

###################################################################################
# Geometry Input Parameters ------------------------------------------------------%
t_plate = 1 # mm
t_ply = 0.125 # mm
l_t,t_t,w_t = 50,0.164,8 # mm
theta_EL_1, theta_EL_2 = (-np.pi/2), np.pi/2
n_overlaps = 4
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
alpha= 53.0* np.pi/180 # Angle of fracture surfaces - determined experimentally and hence is an input specfic to conposite in consideration (for thick ply)
# Num Input Parameters -----------------------------------------------------------%
N_e = 200
N_tau12,N_sigma2 = 50,50 ### IMPORTANT - control speed of script -> initially set to 100 each!
# Loading Vector Input Parameters ------------------------------------------------%
sigma_g_unnormalised = [Alpha,Beta,Gamma] = [100,5,0.4] # unit global loading vector in X,Y,S direction
sigma_g = sigma_g_unnormalised/np.linalg.norm(sigma_g_unnormalised) 
###################################################################################

# Generated Properties -----------------------------------------------------------%
X_T = X_T_UD*(V_f/V_f_UD) # Longitudinal Tensile Strength in MPa
E_1 = E_1_UD*(V_f/V_f_UD) # Young's Modulus in MPa
t_char = (w_t*t_t)/(2*(w_t+t_t)) # Charatceristic Thickness
N_t = int(t_plate/t_ply) # Number of plys in laminate
S_is=np.sqrt(2)*S # in-situ shear strength
Y_is=1.12*np.sqrt(2)*Y_t # in-situ transverse strength
eta=-S/Y_c*np.cos(2*alpha)/np.cos(alpha)**2 # friction coefficient
theta_t = [rnd.uniform(theta_EL_1,theta_EL_2) for i in range(N_t)] # Angle of each ply in laminate
overlap_lenths = list(np.zeros(N_t))
overlap_thicknesses = list(np.zeros(N_t))
for i in range(N_t): # generating N_t sets of n_overlap random overlap lengths and thicknesses
    overlap_lenths[i] = [rnd.uniform(0,l_t/2) for j in range(n_overlaps)]
    overlap_thicknesses[i] = [rnd.uniform(0,2*t_char) for j in range(n_overlaps)]
overlap_lenths = np.array(overlap_lenths)
overlap_thicknesses = np.array(overlap_thicknesses)

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
Q_i_N = [Q_0_to_Q_i(theta,Q_0) for theta in theta_t] # All Q_i matrices in one array
Q_g = np.mean(Q_i_N,axis=0) # Q_g is the weighted average of all Q_i matrices depending on ply thickness (equal here)

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
I = np.reshape([[1,0,0],[0,1,0],[0,0,2]],(3,3))
# T = Transform_Matrix(theta_t[i])
# Q_i = np.linalg.multi_dot([np.linalg.inv(T),Q_0,I,T])
# Both are valid - check NASA document
# Difference due to definition of strain used 
sigma_l = list(np.zeros(N_t)) # Local loading vectors for each ply at theta
for i in range(0,N_t):
    sigma_l[i] = [np.linalg.multi_dot([Q_i_N[i],Transform_Matrix(theta_t[i]),np.linalg.inv(Q_g),np.transpose(sigma_g)])] 
    # generating alpha, beta and gamma for each ply
sigma_l = np.array(sigma_l) ### GIVES LOCAL LOADING VECTORS TO USE TO FIND FAILURE OF EACH PLY (INTERSECTION WITH EACH FAILURE SURFACE)

# Calculating Local Failure Factor -----------------------------------------------%
### Given X_t[i] and a local loading vector simga_i[i], calculate local failure factor m from ITSC
### Need to generate ITSC and find intersection with each local loading vector direction

local_failure_factors = []
for index in range(0,N_t):
    # Generate ITSC_Surface for each ply within the laminate - different set of random overlap lengths and thicknesses for each
    ITSC_Surf, tau12_input, sigma2_s = ITSC_Generation(N_tau12,S,Y_is,eta,N_sigma2,n_overlaps,l_t,E_1,X_T,G_II_c,N_e,
                                                       overlap_lenths[index],overlap_thicknesses[index])

    # Extract x and y values from ITSC_Surface
    x_values = [curve[0] for curve in ITSC_Surf]
    y_values = [curve[1] for curve in ITSC_Surf]

    # Create 3D line plot for each value of sigma2_s
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over the x, y, and sigma2_s values and plot the lines
    for x, y, sigma2 in zip(x_values, y_values, sigma2_s):
        ax.plot(x, y, sigma2) # doesn't show the X=0,Y=max,S=0 point so plotted individually below

    ax.plot(x_values[-1],y_values[-1],sigma2_s[-1], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=2, alpha=0.6)

    # Set axis labels
    ax.set_xlabel('X [MPa]')
    ax.set_ylabel('S [MPa]')
    ax.set_zlabel('Y [MPa]')

    # Set the view
    # ax.view_init(elev=0, azim=0)  # Set elevation to 90 degrees (top view) and azimuth to 0 degrees

    # Remove the title
    ax.set_title('ITSC Failure Surface')

    # Show the plot
    plt.show() # CORRECT
