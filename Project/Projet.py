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

Le but principal de ce script est de chercher le parmaètre optimal pour les temps de sauts du 
processus de poisson
"""

from random import uniform as U
from random import normalvariate as W
from math import log,exp,tan,pi,sqrt,cos,sin
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
    

def naiv_parametrix(T,Lambda,counter):
    #naiv implementation of the forward method for a simple exemple
    X_pred = x0
    TETA = 1
    #
    t = 0
    delta_tau = -(1./Lambda)*log(U(a=0,b=1))
    t += delta_tau
    Var = 0
    if(delta_tau > T):
        counter = counter + 1
        X_new = x0 + b(x0) * T + sig(x0) * sqrt(T) * W(0,1)
        Var = (exp(Lambda * T) * f(X_new))**2
        return exp(Lambda*T)*f(X_new) , counter , Var
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
        Var = (exp(Lambda*T)*f(last_X) * TETA)**2
        return exp(Lambda*T) * f(last_X) * TETA , counter, Var

    


#recherche quadrillée pour le lambda optimal
counter = 0
result_dic = {}
#dictionanire ou seront stoqués les résultats 
Lambda = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
Lambda_2 = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
N = 100000
k = 0
V = 0
N_V = 0
K_V = 0
for i in Lambda:
    for j in range(N):
        u = naiv_parametrix(horizon,i,counter)
        k = k + u[0]
        V = V + u[2]
    result_dic["lambda = " +str(i)]=[i, k / N , V/(N-1) - N/(N-1)*(k / N)**2, \
               "[" + '%.4f'% (k / N - 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+','+'%.4f'% (k/N + 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+ ']']
    k = 0
    V = 0
    N_V = 0
    K_V = 0   



#extraction de la valeur optimale pour notre paramètre
min_var = float("inf")
for val in result_dic.values():
    if val[2] < min_var : 
        L_star = val[0]
        min_var = val[2]
    
    
#Cette partie sert à la descritpion du parmètre
N = 100000
k = 0
V = 0
counter = 0
for i in range(N):
    u = naiv_parametrix(horizon,L_star,counter)
    k = k + u[0]
    counter = u[1]
    V = V + u[2]

exact_value = f(x0)
numerical_value = k / N
Var = V/(N-1) - N/(N-1)*(numerical_value**2)

#partie affichage machine

print("valeurs exacte = " + str(exact_value))

print("notre valeurs optimale trouvée est : "+ str(L_star))
print("valeurs numérique pour ce parmaètre = " + str(numerical_value))
absolute_error = abs(exact_value - numerical_value)
print("erreur absolue = " + str(absolute_error))


if(x0 != 0):
    relative_error = absolute_error / abs(exact_value)
    print("erreur relative = " + str(relative_error*100) + " %")

print("Variance pour notre modèle " + str(Var))

print("\nnotre intervalle de confiance est : [" + "%.4f"%(numerical_value- 1.96*sqrt(Var/N))+","+"%.4f"%(numerical_value + 1.96*sqrt(Var/N))+ "]")

print("\n"+str((float(counter)/N)*100)+"% des estimateurs sont des schémas d'euler à un pas")




#essai païen
#
#def parametrix_Millstein(T,Lambda,counter):
#    #naiv implementation of the forward method for a simple exemple
#    X_pred = x0
#    TETA = 1
#    #
#    t = 0
#    delta_tau = -(1./Lambda)*log(U(a=0,b=1))
#    t += delta_tau
#    Var = 0
#    if(delta_tau > T):
#        counter = counter + 1
#        N = sqrt(T)*W(0,1)
#        X_new = X_pred + T * b(X_pred) + N*sig(X_pred) + (0.5)*(sig(X_pred)**2)*(N**2 - T)
#        Var = (exp(Lambda * T) * f(X_new))**2
#        return exp(Lambda*T)*f(X_new) , counter , Var
#    else :
#        while(t < T):
#            N = sqrt(delta_tau)*W(0,1)
#            X_new = X_pred + delta_tau * b(X_pred) + N*sig(X_pred) + (0.5)*(sig(X_pred)**2)*(N**2 - delta_tau)
#            TETA = TETA*teta(delta_tau,X_pred,X_new) / Lambda 
#            X_pred = X_new
#            delta_tau = -(1./Lambda)*log(U(a=0,b=1)) 
#            t = t + delta_tau
#                       
#        delta = T - t + delta_tau
#        N = sqrt(delta_tau)*W(0,1)
#        last_X = X_pred + delta * b(X_pred) + N*sig(X_pred) + (0.5)*(sig(X_pred)**2)*(N**2 - delta)
#        TETA = TETA*teta(delta,X_pred,last_X) / Lambda 
#        Var = (exp(Lambda*T)*f(last_X) * TETA)**2
#        return exp(Lambda*T) * f(last_X) * TETA , counter, Var


#
#for i in Lambda:
#    for j in range(N):
#        u = parametrix_Millstein(horizon,i,counter)
#        k = k + u[0]
#        V = V + u[2]
#    result_dic["lambda = " +str(i)]=[i, k / N , V/(N-1) - N/(N-1)*(k / N)**2, \
#               "[" + '%.4f'% (k / N - 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+','+'%.4f'% (k/N + 1.96*sqrt((V/(N-1) - N/(N-1)*(k / N)**2)/N))+ ']']
#    k = 0
#    V = 0
#    N_V = 0
#    K_V = 0  
