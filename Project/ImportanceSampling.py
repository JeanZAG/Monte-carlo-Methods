#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:40:03 2018

@author: zagdoun & Desmeuzes

Projet Monte-Carlo : simulation d'une EDS

méthode Forward Parametrix avec grille temporelle iid de loi Bêta

La diffusion Xt est unidimensiopnelle 

on admet que F(Xt) est une vraie martingale avec les coeficient de diffusion(sig()) et de dérive(b())
définient comme ci-dessous

Le but principal de ce script est d'implémenter une méthode de réduction de variance et donc de
chercher les parmaètres optimal pour notre nouveau schéma
"""

from random import uniform as U
from random import normalvariate as W
from math import log,exp,tan,pi,sqrt,cos,sin,isnan
import numpy as np


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
    

def pn(tau_bar,S,T,gamma):
    #fonction auxiliaire pour l'imporance sampling
    aux = 1
    n = S.size
    if (n > 1):
        for i in range(S.size - 1):
            aux = aux * (1-gamma)/(((S[i+1]-S[i])**gamma)*(tau_bar**(1-gamma)))
    res = (1-((T-S[-1])/tau_bar)**(1-gamma))*aux 
    return res


def parametrix_beta(T,gamma,counter,tau_bar):
    #naiv implementation of the forward method for a simple exemple
    X_pred = x0
    TETA = 1
    t = 0
    E = -1*log(U(a=0,b=1))
    delta_tau = tau_bar * exp(-(1-gamma)*E)
    t += delta_tau
    Var = 0
    Sn = np.array([t])
    if(delta_tau > T):
        counter = counter + 1
        X_new = x0 + b(x0) * T + sig(x0) * sqrt(T) * W(0,1)
        Sn = np.array([0])
        P = pn(tau_bar,Sn,T,gamma)
        Var = (f(X_new)/P)**2
        return  f(X_new)/P , Var , counter
    else :
        
        while(t < T):
            X_new = X_pred + delta_tau * b(X_pred) + sqrt(delta_tau)*W(0,1)*sig(X_pred)
            TETA = TETA*teta(delta_tau,X_pred,X_new)
            X_pred = X_new
            E = -1*log(U(a=0,b=1))
            delta_tau = tau_bar * exp(-(1-gamma)*E)
            t = t + delta_tau
            if(t < T):
                Sn = np.append(Sn,t)
                       
        delta = T - t + delta_tau
        last_X = X_pred + delta * b(X_pred) + sqrt(delta)*W(0,1)*sig(X_pred)
        TETA = TETA*teta(delta,X_pred,last_X) 
        P = pn(tau_bar,Sn,T,gamma)
        Var = (f(last_X) * TETA/ P) ** 2
        return f(last_X) * TETA/ P , Var , counter



#recherche quadrillée pour le gamma et le tau bar optimal
counter = 0
result_dic = {}
gamma = [0.7,0.75,0.8,0.85,0.9]
tau_bar = {100,200,500}
N = 100000
k = 0.
V = 0.
for i in gamma:
    for t in tau_bar:
        for j in range(N):
            u = parametrix_beta(horizon,i,counter,5)
            k = k + u[0]
            V = V + u[1]
        result_dic["gamma = " +str(i)]=[i, k / N , V/(N-1) - N/(N-1)*(k / N)**2, \
                   "[" + '%.4f'% (k / N - 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+\
                   ','+'%.4f'% (k/N + 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+ ']', t]
        k = 0
        V = 0


#extraction des paramètres optimaux
min_var = float("inf")
for val in result_dic.values():
    if val[2] < min_var : 
        G_star = val[0]
        min_var = val[2]
        t_star = val[4]
    
    
#some insight for this parameter 
N = 100000
k = 0
V = 0
counter = 0
for i in range(N):
    u = parametrix_beta(horizon,G_star,counter,t_star)
    k = k + u[0]
    counter = u[1]
    V = V + u[2]

exact_value = f(x0)
numerical_value = k / N
Var = V/(N-1) - N/(N-1)*(k / N)**2
print("Exact value = " + str(exact_value))
print("Numerical value = " + str(numerical_value))
absolute_error = abs(exact_value - numerical_value)
print("Absolute error = " + str(absolute_error))


if(x0 != 0):
    relative_error = absolute_error / abs(exact_value)
    print("Relative error = " + str(relative_error*100) + " %")


print("\n"+str((float(counter)/N)*100)+"% of the estimators are one step euler scheme")

print("gamma = " +str(G_star))
print("tau bar = " +str(t_star))