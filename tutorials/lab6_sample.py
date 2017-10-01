# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 21:24:55 2014

@author: srmarsla
"""

#Use module of som and pya

import numpy as np
import pylab as pl

# Have a look at iris.py on the webpage for just using the SOM and k-means to learn about the iris dataset

# This isn't proper normalisation, but will do!
iris = np.loadtxt('iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

target = iris[:,4]

order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

import som

def makesom(x,y):
    # Make and train a SOM
    net = som.som(x,y,train)
    net.somtrain(train,400)
    #return net

    # Store the best node for each training input
    best = np.zeros(np.shape(train)[0],dtype=int)
    for i in range(np.shape(train)[0]):
        best[i],activation = net.somfwd(train[i,:])
    return best
    
def countoverlaps(target,best):
    # Find places where the same neuron represents different classes
    i0 = np.where(traint==0)
    nodes0 = np.unique(best[i0])
    i1 = np.where(traint==1)
    nodes1 = np.unique(best[i1])
    i2 = np.where(traint==2)
    nodes2 = np.unique(best[i2])

    # Lots of ways to do this, but this is neat
    # Other way could be [item in nodes1 for item in nodes0]
    doubles01 = np.in1d(nodes0,nodes1,assume_unique=True)
    doubles02 = np.in1d(nodes0,nodes2,assume_unique=True)
    doubles12 = np.in1d(nodes1,nodes2,assume_unique=True)

    return len(nodes0[doubles01]) + len(nodes0[doubles02]) + len(nodes1[doubles12])

score = np.zeros((4,1))
count = 0
for x in [2,4]:
    for y in [2,4]:
        best = makesom(x,y)
        Noverlaps = countoverlaps(target,best)
        # So a possible score is:
        score[count] = x*y + 10*Noverlaps
        count += 1

print score
# Now pick the best
print np.argmax(score)

# There really isn't any new code for Question 2. If you want me to look at your solution, then feel free to email it to me. 

