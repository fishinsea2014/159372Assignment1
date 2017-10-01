# -*- coding: utf-8 -*-
## {{{ http://code.activestate.com/recipes/578241/ (r1)
from operator import itemgetter, attrgetter
import math
import random
import string
import timeit
from timeit import Timer as t
import numpy as np
import pickle
import os 

# import xlrd


#_-----��ݷ�װ--------------------------
def pakge(i,table):
  index1=table.cell_value(i,1)
  index2=table.cell_value(i,2)
  index3=table.cell_value(i,3)
  index4=table.cell_value(i,4)
  index5=table.cell_value(i,5)
  index6=table.cell_value(i,6)
  index7=float(table.cell_value(i,7))/1000000
  index8=float(table.cell_value(i,8))/1000000
  
  index11=table.cell_value(i+1,1)
  index21=table.cell_value(i+1,2)
  index31=table.cell_value(i+1,3)
  index41=table.cell_value(i+1,4)
  index51=table.cell_value(i+1,5)
  index61=table.cell_value(i+1,6)
  index71=float(table.cell_value(i+1,7))/1000000
  index81=float(table.cell_value(i+1,8))/1000000
  
  index12=table.cell_value(i+2,1)
  index22=table.cell_value(i+2,2)
  index32=table.cell_value(i+2,3)
  index42=table.cell_value(i+2,4)
  index52=table.cell_value(i+2,5)
  index62=table.cell_value(i+2,6)
  index72=float(table.cell_value(i+2,7))/1000000
  index82=float(table.cell_value(i+2,8))/1000000
  
  index13=table.cell_value(i+3,1)
  index23=table.cell_value(i+3,2)
  index33=table.cell_value(i+3,3)
  index43=table.cell_value(i+3,4)
  index53=table.cell_value(i+3,5)
  index63=table.cell_value(i+3,6)
  index73=float(table.cell_value(i+3,7))/1000000
  index83=float(table.cell_value(i+3,8))/1000000
  
  index14=table.cell_value(i+4,1)
  index24=table.cell_value(i+4,2)
  index34=table.cell_value(i+4,3)
  index44=table.cell_value(i+4,4)
  index54=table.cell_value(i+4,5)
  index64=table.cell_value(i+4,6)
  index74=float(table.cell_value(i+4,7))/1000000
  index84=float(table.cell_value(i+4,8))/1000000
  indexs=[
      index1,index2,index3,index4,index5,index6,index7,index8,
      index11,index21,index31,index41,index51,index61,index71,index81,
      index12,index22,index32,index42,index52,index62,index72,index82,
      index13,index23,index33,index43,index53,index63,index73,index83,
      index14,index24,index34,index44,index54,index64,index74,index84
  ]
  indexs1=[]
  for i in indexs:
      indexs1.append(float(i)*1)
  return indexs1
  
#--------------------







pat = [
]
testpat = [
] 

def lodaData(code):
    data = xlrd.open_workbook(str(code)+'.xls')
    table = data.sheets()[0] 
    nrows = table.nrows-10
    for i in range(nrows):
      i=i+1
      indexs=pakge(i,table)
      outputValues=[]
      outputValues.append(table.cell_value(i+5,2))
      outputValues.append(table.cell_value(i+6,2))
      outputValues.append(table.cell_value(i+7,2))
      outputValues.append(table.cell_value(i+8,2))
      outputValues.append(table.cell_value(i+9,2))
  
      index1=table.cell_value(i+4,2)
      outputValue=max(outputValues)
      returnValue=outputValue-index1
      outputValue=-10
      name='xx'
      if returnValue>0:
        outputValue=10
      pat.append([indexs,[outputValue],[name]])


    a=nrows+5 #�Ƶ����һ�����
    for i in range(1):
      i=a
      indexs=pakge(i,table)
      taget2=table.cell_value(i+4,2)
      testpat.append([indexs,[taget2],[name]])
      a=a+1




 #--------------------------------------------
def sigmoid (x):
  return math.tanh(x)
 
def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
   
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)
 
class NN:
  def __init__(self, NI, NH, NO):
    self.ni = NI
    self.nh = NH
    self.no = NO
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    self.wi = [ [0.0]*self.nh for i in range(self.ni) ]
    self.wo = [ [0.0]*self.no for j in range(self.nh) ]
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
 
  def runNN (self, inputs):
    if len(inputs) != self.ni:
      print 'incorrect number of inputs'
    for i in range(self.ni):
      self.ai[i] = inputs[i]
    for j in range(self.nh):
      self.ah[j] = sigmoid(sum([ self.ai[i]*self.wi[i][j] for i in range(self.ni) ]))
    for k in range(self.no):
      self.ao[k] = sigmoid(sum([ self.ah[j]*self.wo[j][k] for j in range(self.nh) ]))
    return self.ao
 
  def weights(self):
    print 'Input weights:'
    for i in range(self.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(self.nh):
      print self.wo[j]
    print ''
 
  def test(self, patterns):
    results, targets = [], []
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.runNN(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str(self.runNN(inputs)).rjust(65), 'Target', p[1], result)
      results+= self.runNN(inputs)
      targets += p[1]
    return results, targets
 
  def sumErrors (self):
    error = 0.0
    for p in pat:
      inputs = p[0]
      targets = p[1]
      self.runNN(inputs)
      error += self.calcError(targets)
    inverr = 1.0/error
    return inverr
 
  def calcError (self, targets):
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2
    return error
 
  def assignWeights (self, weights, I):
    io = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.wi[i][j] = weights[I][io][i][j]
    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        self.wo[j][k] = weights[I][io][j][k]
 
  def testWeights (self, weights, I):
    same = []
    io = 0
    for i in range(self.ni):
      for j in range(self.nh):
        if self.wi[i][j] != weights[I][io][i][j]:
          same.append(('I',i,j, round(self.wi[i][j],2),round(weights[I][io][i][j],2),round(self.wi[i][j] - weights[I][io][i][j],2)))
 
    io = 1
    for j in range(self.nh):
      for k in range(self.no):
        if self.wo[j][k] !=  weights[I][io][j][k]:
          same.append((('O',j,k), round(self.wo[j][k],2),round(weights[I][io][j][k],2),round(self.wo[j][k] - weights[I][io][j][k],2)))
    if same != []:
      print same
 
def roulette (fitnessScores):
  cumalativeFitness = 0.0
  r = random.random()
  for i in range(len(fitnessScores)):
    cumalativeFitness += fitnessScores[i]
    if cumalativeFitness > r:
      return i
       
def calcFit (numbers):  # each fitness is a fraction of the total error
  total, fitnesses = sum(numbers), []
  for i in range(len(numbers)):          
    fitnesses.append(numbers[i]/total)
  return fitnesses
 
# takes a population of NN objects
strs='test!'
def pairPop (pop):
  weights, errors = [], []
  for i in range(len(pop)):                 # for each individual
    weights.append([pop[i].wi,pop[i].wo])   # append input & output weights of individual to list of all pop weights
    errors.append(pop[i].sumErrors())       # append 1/sum(MSEs) of individual to list of pop errors
  fitnesses = calcFit(errors)               # fitnesses are a fraction of the total error

  for i in range(int(pop_size*0.15)):
    strs=str(i).zfill(2)+'1/sum(MSEs)'+str(errors[i]).rjust(15)+str(int(errors[i]*graphical_error_scale)*'-').rjust(20)+'fitness'.rjust(12)+str(fitnesses[i]).rjust(17)+str(int(fitnesses[i]*1000)*'-').rjust(20)
  del pop
  return zip(weights, errors,fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple
   
def rankPop (newpopW,pop):
  errors, copy = [], []           # a fresh pop of NN's are assigned to a list of len pop_size
  #pop = [NN(ni,nh,no)]*pop_size # this does not work as they are all copies of eachother
  pop = [NN(ni,nh,no) for i in range(pop_size) ]
  for i in range(pop_size): copy.append(newpopW[i])
  for i in range(pop_size): 
    pop[i].assignWeights(newpopW, i)                                    # each individual is assigned the weights generated from previous iteration
    pop[i].testWeights(newpopW, i)
  for i in range(pop_size): 
    pop[i].testWeights(newpopW, i)
  pairedPop = pairPop(pop)                                              # the fitness of these weights is calculated and tupled with the weights
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True)   # weights are sorted in descending order of fitness (fittest first)
  errors = [ eval(repr(x[1])) for x in rankedPop ]
  return rankedPop, eval(repr(rankedPop[0][1])), float(sum(errors))/float(len(errors))
 
def iteratePop (rankedPop):
  rankedWeights = [ item[0] for item in rankedPop]
  fitnessScores = [ item[-1] for item in rankedPop]
  newpopW = [ eval(repr(x)) for x in rankedWeights[:int(pop_size*0.15)] ]
  while len(newpopW) <= pop_size:                                       # Breed two randomly selected but different chromos until pop_size reached
    ch1, ch2 = [], []
    index1 = roulette(fitnessScores)                                   
    index2 = roulette(fitnessScores)
    while index1 == index2:                                             # ensures different chromos are used for breeeding
      index2 = roulette(fitnessScores)
    #index1, index2 = 3,4
    ch1.extend(eval(repr(rankedWeights[index1])))
    ch2.extend(eval(repr(rankedWeights[index2])))
    if random.random() < crossover_rate:
      ch1, ch2 = crossover(ch1, ch2)
    mutate(ch1)
    mutate(ch2)
    newpopW.append(ch1)
    newpopW.append(ch2)
  return newpopW
 
graphical_error_scale = 100
max_iterations =1
pop_size = 100
mutation_rate = 0.1
crossover_rate = 0.8
ni, nh, no = 40,7,1
 
def main(code):
  lodaData(code)                    
  # Rank first random population
  pop = [ NN(ni,nh,no) for i in range(pop_size) ] # fresh pop
  pairedPop = pairPop(pop)  #��һ��Ⱥ������������
  rankedPop = sorted(pairedPop, key = itemgetter(-1), reverse = True) # THIS IS CORRECT
  # Keep iterating new pops until max_iterations
  tops, avgs = [], []
  #���ض���
  try:
      saveObj =pickle.load( open('fnn_Obj_'+str(code)+'.pkl', 'r'))
      rankedPop=saveObj[0]
      pop=saveObj[1]
  except IOError:   
      print str(code)+'_init'
  iters = 0
  while iters != max_iterations:
    if iters%1 == 0:
      print 'Iteration'.rjust(150), iters
    newpopW = iteratePop(rankedPop)
    
    rankedPop, toperr, avgerr = rankPop(newpopW,pop)
    iters+=1
  
  saveObj1=[rankedPop,pop]
  fileName='fnn_Obj_'+str(code)+'.pkl'
  
  try:
      os.remove(fileName) 
      print 222
  except WindowsError:  
      print "ɾ��ʧ�ܣ�����"

  output = open(fileName, 'wb')
  pickle.dump(saveObj1, output)  #����ѵ��Ȩ��
  
  # test a NN with the fittest weights
  tester = NN (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  results, targets = tester.test(testpat)
  
  print testpat 
  print code
  print results

  #��һ��������
  rt=[results,targets]
  
  output1 = open('fnn_return'+str(code)+'.pkl', 'wb')
  pickle.dump(rt, output1)  #����Ȩ��

  print 'max_iterations',max_iterations,'\tpop_size',pop_size,'pop_size*0.15',int(pop_size*0.15),'\tmutation_rate',mutation_rate,'crossover_rate',crossover_rate,'ni, nh, no',ni, nh, no
 
def crossover (m1, m2):
  r = random.randint(0, (ni*nh)+(nh*no) ) # ni*nh+nh*no  = total weights
  output1 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
  output2 = [ [[0.0]*nh]*ni ,[[0.0]*no]*nh ]
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
 
def mutate (m):
  # could include a constant to control
  # how much the weight is mutated by
  for i in range(len(m)):
    for j in range(len(m[i])):
      for k in range(len(m[i][j])):
        if random.random() < mutation_rate:
            m[i][j][k] = random.uniform(-2.0,2.0)
   
if __name__ == "__main__":
    codes=["cn_300301"]
    for code in codes:
        for i in range(1):
            main(code)

