#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:40:03 2018

@author: zagdoun

Projet Monte-Carlo : Unbiased simulation of an SDE

Forward Parametrix method

The diffusion in this script is a little more difficult to simulate than in the
previous one.
In deed The fact that f(Xt) is a true martingale is not trivial as before
This scheme is a little less precise than the previous one
"""

from random import uniform as U
from random import normalvariate as W
from math import log,exp,tan,pi,sqrt,cos,sin
import numpy as np


"""parameters"""
s = 0.001
C0 = 0.
C1 = 1.
C3 = 1.
x0 = 1
w = 0.001
counter = 0



def sig(x):
    return s*(sin(w*x)+2)

def dev_sig(x):
    return s*w*cos(x*w)
#marche correctement :  retourne les bonnes valeurs

def dev2_sig(x):
    return - s * w * w * sin(x*w)

def b(x):
    return -x / (x**2+(C1/(3*C3))) * sig(x)

def dev_b(x):
    return dev_sig(x) * -x / (x**2+(C1/(3*C3))) + sig(x) * (-x**2-2*x-(C1/(3*C3))) / ((x**2+(C1/(3*C3))))**2

def f(x):
    return C3*(x**3) + C1*x + C0
    
def a(x):
    return (s*(sin(w*x)+2))**2
    
def dev_a(x):
    return 2 * dev_sig(x) * sig(x)

def dev2_a(x):
    return 2 * dev_sig(x)**2 + 2 * dev2_sig(x) * sig(x)

def Hermite(a,x):
    return -(a**-1)*x

def Hermite2(a,x):
    return Hermite(a,x)**2 - a**-1

def kappa(t,x,y):
    return dev2_a(y) + 2 * dev_a(y) * Hermite(t*a(x),y-x-b(x)*t) \
+ (a(y)-a(x)) * Hermite2(t*a(x),y-x-b(x)*t)
    
def rho(t,x,y):
    return dev_b(y) + (b(y)-b(x)) * Hermite(t*a(x),y-x-b(x)*t)
    
def teta(t,x,y):
    return (1/2) * kappa(t,x,y) - rho(t,x,y)
    

def naiv_parametrix(T,Lambda,counter):
    #naiv implementation of the forward method for a simple exemple
    X_pred = x0
    TETA = 1
    #
    t = 0
    delta_tau = -(1./Lambda)*log(U(a=0,b=1))
    t += delta_tau
#    tem = np.array([])
    if(delta_tau > T):
        counter = counter + 1
        X_new = x0 + b(x0) * T + sig(x0) * sqrt(T) * W(0,1)
        return exp(Lambda * T) * f(X_new) , counter
    else :
        while(t < T):
            X_new = X_pred + delta_tau * b(X_pred) + sqrt(delta_tau)*W(0,1)*sig(X_pred)
#            tem = np.append(tem, X_new)
            TETA = TETA*teta(delta_tau,X_pred,X_new) / Lambda 
            X_pred = X_new
            delta_tau = -(1./Lambda)*log(U(a=0,b=1)) 
            t = t + delta_tau
                       
        delta = T - t + delta_tau
        last_X = X_pred + delta * b(X_pred) + sqrt(delta)*W(0,1)*sig(X_pred)
        TETA = TETA*teta(delta,X_pred,last_X) / Lambda 
        return exp(Lambda*T)*f(last_X) * TETA , counter

    
# testing : praying for it to work §§§§§

N = 100000
k = 0
for i in range(N):
    u = naiv_parametrix(1,1,counter)
    k = k + u[0]
    counter = u[1]

exact_value = f(x0)
numerical_value = k / N
print("Exact value = " + str(exact_value))
print("Numerical value = " + str(numerical_value))
absolute_error = abs(exact_value - numerical_value)
print("Absolute error = " + str(absolute_error))


if(x0 != 0):
    relative_error = absolute_error / abs(exact_value)
    print("Relative error = " + str(relative_error*100) + " %")
    
print("durant les " + str(N) + " itérations il y en a eu " + str(counter)+ " qui ne sont pas passer dans l'approche parametrix")

print(str((float(counter)/N)*100)+"% of the estimator are one step euler scheme")