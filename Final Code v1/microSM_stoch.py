import numpy as np
import matplotlib.pyplot as plt

### The aim of this script is to develop a stochastic model to calculate the strength of a tow/interaction within the composite
### A stochastic model is needed to incorporate the randomness of the actual TBDC structure
### Each INTERACTION modelled is made up of 4 seperate overlaps by the 4 neighbouring tows on the tow being evaluated - creating a 4 overlap interaction

def microSM(n_overlaps,S,l_t,E_1,X_T,G_II_c,N_e,l_o_rnd,t_o_rnd):

### Input paramters are:
    ### Number of overlaps being modelled PER interaction (i.e. model is calculating stregth of the interaction) - set to 4
    ### Shear Strength, Elastic Modulus, Tensile Strength and Fracture Toughness of the tows (S, E_1, X_T, G_II_c)
    ### Tow length - l_t
    ### N_e - number of steps to discretise the strain range
    ### l_o_rnd & t_o_rnd - two sets that create 4 pairs of overlap lengths and thicknesses for each overlap in the interaction (matched sets by indices) 
    ### i.e. l_o_rnd[0] matches with t_o_rnd[0]

    # N_e = 200
    e_max = X_T/E_1 # maximum possible strain of interaction before tow fracture - UTS/Elastic Modulus
    e_vals = np.linspace(0,e_max,N_e) # Discretised strain values

    stress_matrix = np.zeros([n_overlaps,len(e_vals)]) # Creating a matrix to hold stress values (i.e. stress-strain curve as strain values are the same)
    # Matrix will hold the stress values for each (of 4) overlaps, and each array in the matrix is the same length based on the discretised strain values

    for i in range(n_overlaps): # Looping to repeat the following calculations for EACH overlap (4 in total)
        l,t = l_o_rnd[i],t_o_rnd[i] # Setting the length and thickness of the overlap being evaluated as a matched pair from the random lists
        # l_t = l # Not sure if this is valid

        # Calculations are being done based on the exact same code as the determinisitc model - calculating the stress-strain curve for 
        # a singular interaction* composed two tows only! (i.e one out of the 4 neighbouring tows in consideration)
        ### However, when I mention interaction* here, I am not referring to the 4 neighbouring tows interaction, but rather an interaction* between
        ### two tows, hence this must be repeated for each interaction* within the overall interaction

        # Now, the overlap length and thickness for this interaction* is set and calculation are performed

        a = (G_II_c)/(S*l_t)
        b = S*l_t
        c = E_1 * G_II_c

        l_o_crit = 2*np.sqrt((E_1*G_II_c*t)/(S**2)) # Critical length for transition between yielding and cracking

        d = S/(2*E_1*t)
        e = ((2*((np.sqrt((c*t)/(S**2)))/l_t)+1)*np.sqrt((c)/t))/E_1
        f = S/(2*t)
        g = (np.sqrt((c)/t))/E_1

        rig_plast = (b*(np.sqrt(((8*E_1*e_vals*t)/(b))+1)-1))/(4*t) # Stress in the rigid plastic linear region
        plat_bigger = np.sqrt((c)/t) # This is the stress value (constant for each overlap thickness) or the PC-S range (l_o > l_o_crit)

        stress_iteration = np.zeros(len(e_vals)) # Stress(-strain) curve for the current interaction* in consideration

        if l < l_o_crit: # tow pull out - RP-FP-S - yielding
            strain_plat_start = d*l*((l/l_t)+1)
            strain_plat_end = a + d*l*((l/(2*l_t))+1)

            stress_iteration[np.where(e_vals <= strain_plat_end)[0]] = f*l
            stress_iteration[np.where(e_vals <= strain_plat_start)[0]] = rig_plast[np.where(e_vals < strain_plat_start)[0]]

        elif l >= l_o_crit: # tow fracture - RP-PC-S - cracking
            strain_plat_start = e
            strain_plat_end = ((l/l_t)+1)*g

            stress_iteration[np.where(e_vals <= strain_plat_end)[0]] = plat_bigger
            stress_iteration[np.where(e_vals <= strain_plat_start)[0]] = rig_plast[np.where(e_vals < strain_plat_start)[0]]    

        # plt.plot(e_vals,stress_iteration)
        # plt.show()

        stress_matrix[i] = stress_iteration # Fill stress matrix with the stress values for each interaction*
        ### All as in the deterministic model
    
    ave_stress = np.mean(stress_matrix, axis = 0) # Averaging over the 4 interactions* to get a stress-strain curve for the interaction

    # plt.plot(e_vals,ave_stress)
    # plt.ylabel('Longitdudinal Tensile Stress [MPa]')
    # plt.xlabel('Longitudinal Tensile Strain')
    # plt.grid()
    # plt.show()

    ### Alternate solution - find first point of failure, i.e stress at which first interaction* fails at the lowest strain
    # for i in range(len(ave_stress)-1):
    #     if ave_stress[i] > ave_stress[i+1]:
    #         return ave_stress[i]
    #     else:
    #         pass
    ### Alternate solution - block this out to just return max value as standard
    ### From tester file, this is working ass expected

    return np.max(ave_stress)

# def Ave_Stress(S,Num_sim,Num_interactions,l_t,t_t,w_t,E_1_UD,X_T_UD,V_f_UD,V_f,G_II_c,N_e):
#     stresses = [Stoch_Model(Num_interactions,S,l_t,t_t,w_t,E_1_UD,X_T_UD,V_f_UD,V_f,G_II_c,N_e) for i in range(0,Num_sim)]
#     sigma1 = np.average(stresses)

#     return sigma1

### End result is to calculate the stress-strain curve for 4 different overlaps/interactions*, and average them to form the stres-strain 
### curve for the 4 overlap interaction. The maximum of this curve is taken to get the strength of the tow/interaction in consideration.