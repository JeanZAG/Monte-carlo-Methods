#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:44:20 2017

@author: zagdoun
"""

from random import uniform as U
import numpy as np
from math import sqrt,cos,sin,log,pi,exp
import scipy.stats
import sys


#parameters
T = 1
S0 = 100
sig = 0.2
r = 0.05
L = 140
M = 100000
exact_price = 0.05968
epsilon = sys.float_info.epsilon

#simulation of the Black and scholes process
def expn(L):
    u = U(a=0,b=1)
    return -(1./L)*log(u)

def gauss_Box_Muller(m,s):
    R = sqrt(expn(0.5))
    u = U(a=0,b=2*pi)
    return np.array([(R* cos(u))*sqrt(s)+m,(R* sin(u))*sqrt(s)+m])

S = np.array([])
BM_N = np.array([])
for i in range(M/2):
    N = gauss_Box_Muller(0,1)
    s = S0*exp((r-(sig**2)/2)*T+sig*sqrt(T)*N[0])
    s2 = S0*exp((r-(sig**2)/2)*T+sig*sqrt(T)*N[1])
    S = np.append(S,[s,s2])
    BM_N = np.append(BM_N,N)

def payoff(S):
    #payoff function
    P = np.zeros(M)
    for i in range(M):
       if S[i] > L:
           P[i] = 1
    return P

#standard montecarlo estimator 

P = payoff(S)*exp(-r*T)
estimated_price = (1./M)*np.sum(P)
V_bar = ((1./(M-1))*np.sum((P-estimated_price*np.ones(M))**2))
#we do as if we did not knew the exact value of the price


#confidence intervall
#our r.v. is clearly a bernouilli beacause it takes values 1 or 0(and price is a linear transformation of this)
#but since we simulate the exact price we use the above metric only for accuracy comparison
var = exact_price*(1-exact_price)*exp(-2*r*T)

q = scipy.stats.norm.ppf(0.95)
#getting the quantile coresponding to 95% precision

inf = estimated_price - q*sqrt(V_bar/M)
sup = estimated_price + q*sqrt(V_bar/M)
#our true value is indeed in this intervall so we won!


#Now let's try MC estimator by importance sampling translation

#First let's find a zero to the gradient function
#let's try the Newton Raphson algorithm to compute this
def grad(G,teta,P):
    #G is the gaussian vector from wich we calculated the payoff of the associated B&S process
    #P is payoff price (square of indicatrice is indicatrice)
    #P is our final function that we want to estimate
    #indeed the price is just a prportional value of the expectation of the payoff
    #teta is our current value for teta
    g = 0
    for i in range(M):
        g = g + (teta - G[i])*P[i]*exp(-teta*G[i]+(teta**2)/2)*exp(-2*r*T)
    return (1./M)*g
        

def hess(G,teta,P):
    #same parameters
    H = 0
    for i in range(M):
        H = H + (1+ (teta-G[i])**2)*P[i]*exp(-teta*G[i]+(teta**2)/2)*exp(-2*r*T)
    return (1./M)*H

def NewtonRaphson1d(G,P):
    teta = 1
    nablav = grad(G,teta,P)
    nabla2v = hess(G,teta,P)
    while(abs(nablav)>epsilon):
        teta = teta - (nablav / nabla2v) 
        nablav = grad(G,teta,P)
        nabla2v = hess(G,teta,P)
    return teta


T_S = NewtonRaphson1d(BM_N,P)
BM_S = BM_N +np.ones(M)*T_S
P_S = payoff(S0*exp((r-(sig**2)/2)*T)*np.exp(sig*sqrt(T)*BM_S))*exp(-r*T)
MC_price = (1./M)*np.sum(P_S*np.exp(-T_S*BM_N)*exp(-(T_S**2)/2))

V_Sbar = ((1./(M-1))*np.sum((P_S*np.exp(-T_S*BM_N)*exp(-(T_S**2)/2)- MC_price*np.ones(M))**2))

inf_S = MC_price - q*sqrt(V_Sbar/M)
sup_S = MC_price + q*sqrt(V_Sbar/M)

if(sup_S-inf_S< sup - inf and (inf_S<exact_price<sup_S)):
    print("\nknowing that our true value is : "+str(exact_price)+"\n")
    print("\nwe have an much more precise intervall than with the previous method\n")
    print("indeed our curent intervall wiht the new MC method is ["\
                                +str(round(inf_S,5))+','+str(round(sup_S,5))+']')
    print("and the value found with this method is : "+str(round(MC_price,5)))
    print("And our previous intervall of confidence given by LLN was: ["\
                                +str(round(inf,5))+','+str(round(sup,5))+']')
    print("and the value found with LLN method is : "+str(round(estimated_price,5)))
    print("\nboth are accurate with approx 0.95 probability (in fact less since this bound is asymptotically true..)")

