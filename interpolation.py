#a) simple routine to perform linear interpolation on x-y data

#import matrix solver LU

import LU as m

#linear interpolation -> join points with a straight line

import matplotlib.pyplot as plt

import matplotlib.axes as ax

import numpy as np

#x,y coordinates for each point as lists

#b) cubic spline interpolation using natural boundary condition f''_0 = f''_n = 0

#use matrix solver from Q2 as outside routine

# form the matrix from the simultaneous equations
# use eqn 4.15

#dimensions of matrix n-1 x n-1, n is the number of points

#can form the v vector where f_i = y points

def interp(xdata,ydata):

    matrix = []

    v = []

    # i 1 to n-1, python i = 0 is first entry but here we need to go from i = 1 to n-1

    n = len(xdata)

    z = 0

    j = 0

    # n points n - 1 equations so n - 1 entries in matrix

    #np.zeros(,) - nxn

    while z < n - 2: # n - 1 because first and last data points positions will not have outer data      
                     # to use for calcs
        matrix.append([])

        v.append(0)
        
        while j < n - 2:
            
                matrix[z].append(0) # makes all entries= 0
                
                j += 1
                
        j = 0
        
        z += 1

    #empty matrix with correct dimensions works

    #now filling in the matrix
    # every i move to next coloumn and change the zeros to the coefficients

    i = 1

    j = 0

    offset = 0 # each row will have coefficients move one coloumn to right

    # MAKE THIS A TRIDAGANOL MATRIX REMOVE ALL F''_0 TERMS SEE WRITTEN PAD NOTE

    while i <= n - 2:
        
        #starting position is at 0 for list so use i - 1 to place numbers in matrix

        #need another counter for coefficients

        co = 1

        #need another statement for the last two rows
        
        while (j + offset) <= n - 2: #as j starts from 0?

            #using formula 4.15

            if co == 1 and i > 1:
                
                matrix[i-1][j + offset] = (xdata[i]-xdata[i-1])/6

            elif co == 1 and i == 1:

                matrix[i-1][j + offset] = (xdata[i+1]-xdata[i-1])/3

                offset = -1

            elif co == 2 and (i < n - 1):
                
                matrix[i-1][j + offset] = (xdata[i+1]-xdata[i-1])/3

            elif (co == 3) and (i < n - 2):

                matrix[i-1][j + offset] = (xdata[i+1]-xdata[i])/6

           # check whether entering last two rows

            co += 1

            j += 1

        j = 0

        offset += 1

        i += 1

    # compute v vector

    i = 1

    while i <= n - 2:

        v[i-1] = ( (ydata[i+1] - ydata[i])/(xdata[i+1] - xdata[i]) ) - ( (ydata[i] - ydata[i-1])/(xdata[i] - xdata[i-1]) )
        
        i += 1

    #once solved for f''_n then plot the cubic spline f(x) at as many points as you like
    #so loop over x values and plot f(x)

    #use matrix solver to get f''_n values as a vector

    #implement boudary conditions

    # here matrix is always an upper matrix so can set L = I

    # fdash is f derivatives

    L = m.LU(matrix)[0]

    U = m.LU(matrix)[1]

    fdash = m.solvex(L,U,v)

    # boudary conditions:

    fdash2 = [0] # sets f''_0 = 0

    fdash2 += fdash

    fdash2.append(0) # sets f''_n = 0

    # set number of points to plot cubic spline

    dx = 0.001

    ddx = dx

    y = []

    x = []

    # spline exists between every two points
    # must loop through values between each x points and plot spline for those then move next pair

    # plot from x[0] to x[-1]

    i = 0

    while i < n-1:

        # plot between points
        
        while (xdata[i] + dx) < xdata[i+1]:

            A = (xdata[i+1] - (xdata[i] + dx))/(xdata[i+1] - xdata[i])

            B = 1 - A

            C = (((A**3 - A) * (xdata[i+1] - xdata[i])**2))/6

            D = (((B**3 - B) * (xdata[i+1] - xdata[i])**2))/6
            
            y.append( (A*ydata[i]) + (B*ydata[i+1]) + (C*fdash2[i]) + (D*fdash2[i+1]) )

            x.append(xdata[i] + dx)

            dx += ddx

        dx = 0

        i += 1

    return np.array(x),np.array(y)

def linearinter(xdata,ydata):

    # write out as code from notes

    i = 0

    f = []

    dx = 0 # dx is increment, add value each time to it

    xx = []

    while i < (len(xdata) - 1):

        while (xdata[i] + dx) < xdata[i+1]:

            f.append( ((xdata[i+1]-(xdata[i] + dx))*ydata[i] + ((xdata[i] + dx) - xdata[i])*ydata[i+1])/(xdata[i+1]-xdata[i]) )

            xx.append(xdata[i] + dx)
            
            dx += 0.01

        dx = 0

        i += 1
        
    # return the data then plot outside this function

    return xx, f
    
##    plt.scatter(xx,f, s= 2)
##    
##    plt.scatter(xdata,ydata,s=6)
##    
##    plt.xlabel('x')
##    
##    plt.ylabel('y')
##    
##    plt.title('Linear Interpolation')
##    
##    plt.show()


