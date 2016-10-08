# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:10:54 2016

@author: Sayonsom Chanda
"""

from pprint import pprint
import numpy as np



def forward_elimination(L, b):
    """
    Calculates the forward part of Equation required to solve Ax=b.
    """
    y = np.zeros(len(b))
    for i in xrange(len(b)):
        if i == 0:
            y[i] = b[0]/L[0][0]
        else:
            y[i]=b[i]
            for j in range(i):
                y[i]=y[i]-(L[i][j]*y[j])
            y[i] = y[i]/L[i][i]
    return y

def backward_substitution(U,y):
    x = np.zeros(len(y))
    limit = len(y)-1
    for i in range(limit,-1,-1):
        if i==limit:
            x[i]=y[limit]
        else:
            
            x[i] = y[i]
            for j in range(limit,i,-1):
                x[i]=x[i]-(U[i][j]*x[j])

    return x
            
        

def matrixMul(A, B):
    TB = zip(*B)
    return [[sum(ea*eb for ea,eb in zip(a,b)) for b in TB] for a in A]
 
def p(M):
    """Creates the pivoting matrix for m."""
    x = len(M) #-1    
    P=np.eye(x)
    print P
    for j in range(x):
        print ("ITERATION NO.", j)
        for k in range(j,x):
            Sum = 0
            for i in range(j):
                Sum = Sum + M[i][j]
                print Sum
            M[k][j] = M[k][j]-Sum
            print(M[k][j])
            
        if j==len(M):
            break;
        M=abs(M)
        Y=np.zeros(len(M))
        I=np.zeros(len(M))
        for jj in range(len(M)):
            Y[jj] = M[j:,j].max()
            
        
        I = M.argmax(0)
        print ("This is I : ",I)
        I=I+j
        print I
#        print ("j = ", j)
#        print ("This is I after adding j : ",I)
#        temp=M[j][:]
#        print ("temp = ",temp)
##        print M[j][:]
#        print ("EMM 1",M[I[j]][:])
#        M[j][:] = M[I[j]][:]
#        print ("EMM J",M[j,:])
#        M[I[j]][:] = temp
        print("eejay", I[j])
        temp=M[j,:]
        M[j,:] = M[I[j],:]
        M[I[j],:] = temp
        
        
        
        temp=P[j,:]
        P[j,:] = P[I[j],:]
        P[I[j],:] = temp
        
        for ku in range(j+1,x):
            Sum = 0
            print("this point reached!")
            for iii in range(j-1):
                Sum = Sum + np.dot(M[j,iii],M[iii,ku])
            M[j,ku] = (M[j,ku]-Sum)/M[j,j]
            print ("New M", M)
    
    return P, M
    

        
    
    
    
    
 
def lu(A):
    """
    Decomposes a nxn matrix A by PA=LU and returns L, U and P using Crout 
    Algorithm
    """
    n = len(A)
    L = [[0.0] * n for i in xrange(n)]
    U = [[0.0] * n for i in xrange(n)]
    P = A
    A2 = A #matrixMul(P, A)
    for j in xrange(n):
        U[j][j] = 1
        for i in xrange(j, n):  # starting at L[j][j], solve j-th column of L
            alpha = float(A2[i][j])
            for k in xrange(j):
                alpha -= L[i][k]*U[k][j]
            L[i][j] = alpha
        for i in xrange(j+1, n):# starting at U[j][j+1], solve j-th row of U
            tempU = float(A2[j][i])
            for k in xrange(j):
                tempU -= L[j][k]*U[k][i]
#            if int(L[j][j]) <= 10e-20:
#                    L[j][j] = 10e-
            U[j][i] = tempU/L[j][j]
    return (L, U, P)




    
def croutinverse(A):
    
    """
    This function will calculate the inverse of a square invertible matrix A,
    using Crout method for finding out Lower and Upper Triangular Matrices. 
    
    This can be set up like 'n' number of Ax=b problems with different b 
    matrices and the x matrix becomes the nth column in the inverse matrix. 
    

    
    """
    
    
    #print "Doing Crout Inverse with", A
    Ainv=np.zeros((len(A),len(A)), dtype=float)
    #Ainv.fill(0)
    Bvir = np.identity(len(A))
    #print Bvir
    for i in range(len(A)):
        B=Bvir[:][i]
        
        lower,unitUpper, pivot =lu(A)
        yFwdSolved = forward_elimination(lower,B)
        xBkwdSolved = backward_substitution(unitUpper,yFwdSolved)
        xBkwdSolved = np.asarray(xBkwdSolved)
    
        Ainv[i][:] = xBkwdSolved 

        
    
    
    return Ainv



def croutsolution(A,B):
    return matrixMul(croutinverse(A),B)