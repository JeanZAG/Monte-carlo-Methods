#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:40:03 2018

@author: zagdoun & Desmeuzes

Projet Monte-Carlo : simulation d'une EDS

méthode Forward Parametrix avec grille temporelle iid exponentielle

La diffusion Xt est unidimensiopnelle 

on admet que F(Xt) est une vraie martingale avec les coeficient de diffusion(sig()) et de dérive(b())
définient comme ci-dessous

Le but principal de ce script est d'afficher les diffusions à partir de deux schémas d'approximations différent:
La methode du schéma d'euler classique et la méthode de notre projet : parametrix


"""

from random import uniform as U
from random import normalvariate as W
from math import log,exp,tan,pi,sqrt,cos,sin
import numpy as np
import matplotlib.pyplot as plt


"""parameters"""
s = .1
C0 = 0.
C1 = 1.
C3 = 1.
x0 = 1
w = .1
counter = 0
horizon = 1


def sig(x):
    return s*(sin(w*x)+2)

def dev_sig(x):
    #dérivée première de sig()
    return s*w*cos(x*w)

def dev2_sig(x):
    #dérivée seconde de sig()
    return - s * w * w * sin(x*w)

def b(x):
    return (-x / (x**2+(C1/(3*C3)))) * (sig(x)**2)

def dev_b(x):
    return (2* x**2 * sig(x)**2 - (sig(x)**2 + 2*x*dev_sig(x)*sig(x)) * (x**2 +(C1/(3*C3)))) / (x**2 +(C1/(3*C3)))**2

def f(x):
    return C3*(x**3) + C1*x + C0
    
def a(x):
    return (s*(sin(w*x)+2))**2
    
def dev_a(x):
    return 2 * dev_sig(x) * sig(x)

def dev2_a(x):
    return 2 * dev_sig(x)**2 + 2 * dev2_sig(x) * sig(x)

def Hermite(a,x):
    #calcul du polynôme d'Hermite de degré 1
    return -(a**-1)*x

def Hermite2(a,x):
    #calcul du polynôme d'Hermite de degré 2
    return Hermite(a,x)**2 - a**-1

def kappa(t,x,y):
    #fonction auxiliaire de téta
    return dev2_a(y) + 2 * dev_a(y) * Hermite(t*a(x),y-x-b(x)*t) \
+ (a(y)-a(x)) * Hermite2(t*a(x),y-x-b(x)*t)
    
def rho(t,x,y):
    #fonction auxiliaire de téta
    return dev_b(y) + (b(y)-b(x)) * Hermite(t*a(x),y-x-b(x)*t)
    
def teta(t,x,y):
    #definition de la fonction de poids teta cf poly de presentation
    return (1/2) * kappa(t,x,y) - rho(t,x,y)

def parametrix_plot(T,Lambda,counter):
    #modification de naive_parmetrix pour l'affichage
    X_pred = x0
    TETA = 1
    t = 0
    delta_tau = -(1./Lambda)*log(U(a=0,b=1))
    t += delta_tau
    Var = 0
    bool = True
    if(delta_tau > T):
        counter = counter + 1
        X_new = x0 + b(x0) * T + sig(x0) * sqrt(T) * W(0,1)
        Var = (exp(Lambda * T) * f(X_new))**2
        return exp(Lambda * T) * f(X_new)  , Var , counter, bool, X_new
    else :
        while(t < T):
            X_new = X_pred + delta_tau * b(X_pred) + sqrt(delta_tau)*W(0,1)*sig(X_pred)
            TETA = TETA*teta(delta_tau,X_pred,X_new) / Lambda 
            X_pred = X_new
            delta_tau = -(1./Lambda)*log(U(a=0,b=1)) 
            t = t + delta_tau
                       
        delta = T - t + delta_tau
        last_X = X_pred + delta * b(X_pred) + sqrt(delta)*W(0,1)*sig(X_pred)
        TETA = TETA*teta(delta,X_pred,last_X) / Lambda 
        Var = (exp(Lambda*T)*f(last_X) * TETA) ** 2
        bool = False
        return exp(Lambda*T)*f(last_X) * TETA , Var ,counter, bool , last_X
    
    
def one_step_euler(T):
    #focntion du schéma d'euler à un pas de temps
    X_new = x0 + b(x0) * T + sig(x0) * sqrt(T) * W(0,1)
    Var = f(X_new)**2
    return  f(X_new), Var, X_new


N = 100000
k = 0
V = 0
counter = 0
A = np.array([])
B = np.array([])
C = np.array([])
D = np.array([])
K = 0
for i in range(N):
    u = parametrix_plot(horizon,0.1,counter)
    E = one_step_euler(horizon)
    D = np.append(D,E[0])
    k = k + u[0]
    counter = u[2]
    V = V + u[1]
    if u[3] == False:
        A = np.append(A,u[0])
    else : 
        B = np.append(B,u[0])
    C = np.append(C,u[0])

    
exact_value = f(x0)
numerical_value = k / N
Var = V/(N-1) - N/(N-1)*(numerical_value**2)
print("Exact value = " + str(exact_value))
print("Numerical value = " + str(numerical_value))
absolute_error = abs(exact_value - numerical_value)
print("Absolute error = " + str(absolute_error))

#partie affichage machine pour les figure 

plt.figure(figsize=(8,6))
plt.hist(A,alpha=0.3, bins = 14)
plt.hist(B,alpha=0.3,bins = 14)
plt.legend(['1st poisson jump < ' + str(horizon), 'One step euler scheme'], loc='upper right')


if(x0 != 0):
    relative_error = absolute_error / abs(exact_value)
    print("Relative error = " + str(relative_error*100) + " %")

print("With a variance of " + str(Var))

print("\nour confidence bound is : [" + str(numerical_value- 1.96*sqrt(Var/N))+','+str(numerical_value + 1.96*sqrt(Var/N))+ ']')

print("\n"+str((float(counter)/N)*100)+"% of the estimators are one step euler scheme")

plt.show()

plt.figure(figsize=(8,6))
plt.hist(C,bins=18,alpha =0.3)
plt.hist(D,bins=18,alpha =0.3)
plt.legend(['Parametrix dsitribution' , 'One step euler scheme'], loc='upper right')

plt.show()
