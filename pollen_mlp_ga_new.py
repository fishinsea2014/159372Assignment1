__author__ = 'QSG'

import random
import numpy as np
from operator import itemgetter

import pylab
import testData
#mlp4ga.py is modified mlp class which fit the requirement of the GA
import mlp4ga

#setup program constant
# graphical_error_scale=100
max_iterations=10
pop_size=100
mutation_rate=0.1
crossover_rate=0.8


pat=testData.train()
patt=testData.traint()


def fitness(errors):
    total=sum(errors)
    fitnesses=[]
    for i in range (len(errors)):
        fitnesses.append(errors[i]/total)
    return  fitnesses

#make a population of mlp objects
def makePops(pop):
    weights=[]
    errors=[]
    for i in range(len(pop)):
        weights.append([pop[i].wi.tolist(),pop[i].wo.tolist()])
        errors.append(pop[i].sumErrors())
    fitnesses=fitness(errors)
    # print 'fitnesses: ', fitnesses
    del pop
    print 'weights------ ',weights
    strPop=zip(weights,errors,fitnesses)

    print 'errors------',strPop[2]
    # print 'str pops',strPop
    return strPop

def roulette(fitnessScores):
    cumalativeFitness=0.0
    r=np.random.random()
    for i in range(len(fitnessScores)):
        cumalativeFitness+=fitnessScores[i]
        if cumalativeFitness>r:
            return i

def crossover(m1,m2):
    print 'child1',m1
    print 'child2',m2
    ni=len(m1[0])
    nh=len(m1[0][0])
    no=1

    r = random.randint(0, (ni*nh)+(nh*no) ) # ni*nh+nh*no  = total weights
    output1 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
    output2 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
    print 'output1:',output1
    print 'output2:',output2
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            for k in range(len(m1[i][j])):
                if r >= 0:
                  output1[i][j][k] = m1[i][j][k]
                  output2[i][j][k] = m2[i][j][k]
                elif r < 0:
                  output1[i][j][k] = m2[i][j][k]
                  output2[i][j][k] = m1[i][j][k]
    r -=1
    return output1, output2

def mutate(m):
    # could include a constant to control
  # how much the weight is mutated by
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        if random.random() < mutation_rate:
            m[i][j][k] = random.uniform(-2.0,2.0)

def iteratePop(rankedPop):
    rankedWeights=list([item[0]for item in rankedPop])
    fitnessScores=[item[-1] for item in rankedPop]
    print 'rankedWeights: ', type(rankedWeights[0][0])
    newpopW=[eval(repr(x)) for x in  rankedWeights[:int(pop_size*0.15)]]
    while len(newpopW)<=pop_size:
        ch1=[]
        ch2=[]
        index1=roulette(fitnessScores)
        index2=roulette(fitnessScores)
        while index1==index2:
            index2=roulette(fitnessScores)
        ch1.extend(eval(repr(rankedWeights[index1])))
        ch2.extend(eval(repr(rankedWeights[index2])))
        if np.random.random()<crossover_rate:
            ch1,ch2=crossover(ch1,ch2)

        mutate(ch1)
        mutate(ch2)
        newpopW.append(ch1)
        newpopW.append(ch2)
    return newpopW

def rankPop(newpopW):
    errors, copy = [], []
    pop = [mlp4ga.mlp(pat,11,patt) for i in range(pop_size) ]
    print 'pop size:',len(newpopW)
    for i in range(pop_size): copy.append(newpopW[i])
    for i in range(pop_size):
        pop[i].assignWeights(newpopW, i)                                    # each individual is assigned the weights generated from previous iteration
        pop[i].testWeights(newpopW, i)
    for i in range(pop_size):
        pop[i].testWeights(newpopW, i)
    pairedPop = makePops(pop)                                              # the fitness of these weights is calculated and tupled with the weights
    rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)   # weights are sorted in descending order of fitness (fittest first)
    errors = [ eval(repr(x[1])) for x in rankedPop ]
    return rankedPop, eval(repr(rankedPop[0][1])), float(sum(errors))/float(len(errors))

#rank first random population
# p1=mlp4ga.mlp(pat,11,pattest)
pop=[mlp4ga.mlp(pat,11,patt)for i in range(pop_size)]
# print 'input weight: ',pop[0].wi

pops=makePops(pop)
# strPop=makePops(pop)
# print 'weights:',pops[0]
rankedPop = sorted(pops, key = itemgetter(-1), reverse = True)

iters=0
tops=[]
avgs=[]
while iters !=max_iterations:
    childPop=iteratePop(rankedPop)
    print 'childPop',len(childPop)
    rankedPop,toperr,avgerr=rankPop(childPop)
    tops.append(toperr)
    avgs.append(avgerr)
    iters+=1

#test
tester=mlp4ga.mlp(testData.test(),11,testData.testt())
fittestWeights=[x[0] for x in rankedPop]
tester.assignWeights(fittestWeights,0)
result,targets=tester.test()


