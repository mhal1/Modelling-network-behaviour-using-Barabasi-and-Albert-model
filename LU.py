#crouts method LU decomposition

import numpy as np

def ab(A,B): # N x N multiplication
    #loopthrough A frist row (multiplying elements with B elements n summing) though to lats row
    number_of_coloumns = len(A[0])
    number_of_rows = len(A)
    #setup matrix C as know dimensions then can just add in elements
    C = []
    i = 0
    j = 0
    k = 0
    while i < number_of_rows:
        C.append([])
        h = 0
        while j < number_of_coloumns:
            
            value = 0
            
            while k < number_of_coloumns:
                value += A[i][k]*B[k][h]
                k += 1
            h += 1
            k = 0 
            C[i].append(value)
            j += 1
            
        j = 0
        i += 1

    return C

def det(U):
    
    #mutiply the diagonals
    i = 0
    det = 1
    number_of_rows = len(U)
    while i < number_of_rows:
        det = det*U[i][i]
        i += 1

    return det

#a+b) want N x N matrix returned as U and L ( without Ls diagonal)

# setup matrix as nested list
# each nested list represents a row
# each entry in nested list is part of a coloumn

def LU(matrix):

    #first get dimensions of matrix and make sure it is N x N

    matrix_coloumns = len(matrix[0])
    matrix_rows = len(matrix)

    #if matrix_coloumns != matrix_rows:

    #alpha = al
    #al ii = 1 

    # form L and U matrices and then combine them

    L = []
    U = []

    #python first entry of list index = 0

    i = 0
    j = 0

    #setup L and U as zero matrices and then edit their entires

    #done RUN THIS IN A NEW MODULE TO CHECK IT SETS EM UP (q2matrixLUinitiationcheck.py)

    while i < matrix_rows:
        L.append([])
        U.append([])
        while j < matrix_coloumns:
            if i == j:
                L[i].append(1) #makes all alphaii = 1
            else:
                L[i].append(0)
            U[i].append(0)
            j += 1
        j = 0
        i += 1

    #fill in these using 3.12 and 3.13 in notes

    #first solve for U
    # j = 1,2,3...,N.
    # i = 1,2,3...,j.

    j = 0 # j is the coloumn number
    i = 0 # i is the row number

    # BOTH L AND U NEED TO BE PRODUCED SIMULTANEUOSLY

    #implement equation 3.12 and 3.13
    #this will solve for U and L

    # take first row of L which should be 1,0,0.....,0
    # matrix multiplication with each coloumn of U, working out values as go along
    # use this to work out the values of U
     
    N = matrix_coloumns

    while j <= N-1: #loop from coloumn j = 0 to j = N-1 check whether need <= or < sign

        #implement sum for cross terms

        #upper terms

        while i <= j:
            
            k = 0
        
            sumU = 0

            #sum for U terms
            
            while k <= (i - 1):
            
                sumU += L[i][k]*U[k][j]
            
                k += 1


            if i == 0:
            
                sumU = 0

            U[i][j] = matrix[i][j]  - sumU
            
            i += 1

        #lower terms
                
        i = j + 1
        
        while i <= N - 1:
            
            k = 0
        
            sumL = 0

            #sum for L terms
            
            while k <= (j - 1):
            
                sumL += L[i][k]*U[k][j]
            
                k += 1

            if j == 0:
            
                sumL = 0

            L[i][j] = (1/(U[j][j]))*(matrix[i][j]  - sumL)

            i += 1
        

        i = 0

        j += 1

    return L, U


def printmatrix(matrix):

    
    print("\n"+ "Matrix:" + "\n")

    for i in range(len(matrix)):
        
        print(matrix[i])

#c + d) solves matrix equation LUx=b FOR X using forward and back substitution

def solvex(L,U,b):

    N = len(L) - 1 # as len gives count of components whereas we need N to refer from 0

    #form x and y vector with N components

    x = []

    y = []

    for c in range(N+1):
        
        x.append(0)

        y.append(0)

    #forward substitution
    
    i = 1 #need from second entry

    y[0] = b[0]/L[0][0]

    j = 0

    while i <= N: #loop i = 1 to N

        sumay = 0
        
        while j <= i - 1:
            
            sumay += L[i][j]*y[j]

            j += 1

        j = 0
        
        y[i] = (1/L[i][i])*(b[i] - sumay)

        i += 1

    #backward substitution

    x[N] = y[N]/U[N][N]

    i = N - 1

    while i >= 0:

        j = i + 1

        sumax = 0

        while j <= N:

            sumax += U[i][j]*x[j]

            j += 1

        j = 0

        x[i] = (1/U[i][i])*(y[i] - sumax)

        i -= 1

    return x
