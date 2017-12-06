#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:44:16 2017

@author: zagdoun
"""

from random import uniform as U
from math import log,exp,tan,pi,sqrt,cos,sin
import matplotlib.pyplot as plt
import numpy as np


#simulation d'une bernoulli
def bern(p=0.5):
    u = U(a=0,b=1)
    if u < p:
        return 1
    else:
        return 0
    
print("la valuers de la bernouilli pour un tirage est : "+ str(bern()))


#simulation d'une binomiale
def bin(n,p=0.5):
    sum = 0
    for i in range(n):
        sum += bern(p)
    return sum

print("la valuers de la binomiale pour un tirage est : "+ str(bin(50)))

#clacul de la moyenne empirique(pour n_sample = 1000)
Sn = 0
for i in range(1000):
    Sn += bin(50)
X_bar = (1/1000.)*Sn

print("\n La moyenne empirique est : " +str(X_bar))
print("sachant que la vrai moyenne deverais être = 25 \n")
#on vérifie par ce biais la loi des grands nombres


Sn = 0
for i in range(1000):
    Sn += (bin(50)-X_bar)**2
V_bar = (1/1000.)*Sn

print("La variance empirique est : " +str(V_bar))
print("sachant que la vrai variance deverais être = " +str((1/4.)*50) + "\n")
# de même loi des grands nombres est vérifiée

#simulation d'exponentielles
def expn(L):
    u = U(a=0,b=1)
    return -(1./L)*log(u)


V = []
for i in range(10000):
    V.append(expn(1))

x = range(10000)
true_exp=[]  
for i in range(10000):
    true_exp.append(1*exp(-np.array(x)[i]))


plt.hist(V,100,normed=1)
plt.title("simulation loi exponentielle")
plt.show()

def cauchy(L):
    u = U(a=0,b=1)
    return L*tan(pi*(u-0.5))

C = []
for i in range(10000):
    C.append(cauchy(1))



plt.hist(C,500,[-12,12])
plt.title("simulation loi de Cauchy")
plt.show()

def inv_gauss(x):
    c0 = 2.515517; c1 = 0.802853; c2 = 0.010328
    d1 = 1.432788; d2 = 0.189269; d3 = 0.001308
    if x > 0.5:
        sgn = 1
        x = 1.-x
    else:
        sgn = -1
    t = sqrt(-2. * log(x))
    return sgn * (t-((c2*t+c1)*t+c0)/(1.+t*(d1+t*(d2+d3*t))))

def gauss(m,s):
    u = U(a=0,b=1)
    return inv_gauss(u)*sqrt(s)+m

N = []
for i in range(10000):
    N.append(gauss(0,1))
    
plt.hist(N,100,[-4,4])
plt.title("simulation de loi normale par approximation de la fonction de rep inv")
plt.show()

#3.2 Box muller method

def gauss_Box_Muller(m,s):
    R = sqrt(expn(0.5))
    u = U(a=0,b=2*pi)
    return np.array([(R* cos(u))*sqrt(s)+m,(R* sin(u))*sqrt(s)+m])

BM_N = np.array([])
for i in range(5000):
    BM_N = np.append(BM_N,gauss_Box_Muller(0,1))
    
plt.hist(BM_N,100,[-4,4])
plt.title("simulation loi normale par methode de Box Muller")
plt.show()


#3.3 Methode du rejet
def reject_gamma(a):
    u = U(0,1)
    if u < (exp(1)/(a+exp(1))):
        Y = (((a+exp(1))/exp(1))*u)**(1./a)
    else: 
        Y = -log((1-u)*((a+exp(1))/a*exp(1)))
    if Y < 1:
        q = exp(-Y)
    if Y> 1 :
        q = Y**(a-1)
    else :
        q = 0
    v = U(0,1)
    if v <= q:
        return Y
    else:
        return reject_gamma(a)
    
    
RG = np.array([])
for i in range(5000):
    RG = np.append(RG,reject_gamma(1))
    
plt.hist(RG,100,normed=1)
plt.title("simulation de loi gamma par methode de rejet")
plt.show()
N = 10000
T = 100.
h= T/N


#Mouvement brownien
t = sqrt(h)*np.ones(10000)

Increments = t * BM_N


B = np.cumsum(Increments)
x = np.linspace(0,50, num = B.size)
plt.plot(x,B)
plt.title("simulation du mouvement brownien")
plt.show()

#Loi forte des grands nombres pour le mouvement brownien
y = np.array([0])

y = np.append(y,1/x[1:])


plt.plot(B*y)
plt.title("Loi des grands nombre pour le mouvement brownien")
plt.show()


