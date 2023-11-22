def Intersection_Finder(load_vector,x_values,y_values,sigma2_s):
    ### Script should find the intersection for each ply/interaction considered -> the inputs are the load vector 
    # and failure surface specific to ply
    # Parametrise the loading vector line from the load vector inside the sigma_l array
    a,b,c = abs(load_vector[0][0]),abs(load_vector[0][2]),abs(load_vector[0][1]) 
    # Load vector given in X,Y,S but co-ords given in X,S,Y hence x,y,z for load line reflects X,S,Y nature
    Intersection = False
    parametrised_distance = 1
    x_vals = []
    y_vals = []
    z_vals = []
    while Intersection is False:
        x = a*parametrised_distance # X
        y = b*parametrised_distance # S
        z = c*parametrised_distance # Y

        x_vals += [x]
        y_vals += [y]
        z_vals += [z]

        # Run a checking function on the current x,y,z co-ordinate - if intersected, output Intersection = True, else pass
        # Convert x_values, y_values and sigma2_s into 3d co-ordinates - maybe just list co-ords and check intersection?
        for row_ind in range(0,len(sigma2_s)-1): # filling rows with co-ordinates for each value of sigma2_s
            for col_ind in range(0,len(x_values[0])): # filling each column
                co_ord = [x_values[row_ind][col_ind],y_values[row_ind][col_ind],sigma2_s[row_ind]]
                if x>= co_ord[0] and y >= co_ord[1] and z >= co_ord[2]:
                    Intersection = True
                    break
                if x >= 0 and y>= 0 and z >= sigma2_s[-1]:
                    Intersection = True
                    break

        if Intersection == False:
            parametrised_distance += 0.2

    return x_vals,y_vals,z_vals,parametrised_distance


