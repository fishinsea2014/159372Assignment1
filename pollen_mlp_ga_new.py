__author__ = 'QSG'

import numpy as np
from operator import itemgetter
from random import *
import pylab
import testData
#mlp4ga.py is modified mlp class which fit the requirement of the GA
import mlp4ga

#setup program constant
# graphical_error_scale=100
max_iterations=40
pop_size=100
mutation_rate=0.1
crossover_rate=0.8
ni,nh,no=4,6,1

pat=testData.test()
pattest=testData.testt()



#make a population of mlp objects
def makePops(pop):
    weights=[]
    errors=[]
    for i in range(len(pop)):
        weights.append([pop[i].wi,pop[i].wo])
        errors.append(pop[i].sumErrors())
    fitnesses=calcFit(errors)


#rank first random population
# p1=mlp4ga.mlp(pat,11,pattest)
pop=[mlp4ga.mlp(pat,11,pattest)for i in range(pop_size)]


pops=makePops(pop)
# strPop=makePops(pop)