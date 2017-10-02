__author__ = 'QSG'

import irisData
pat=irisData.pat()
testpat=irisData.testpat()


from operator import itemgetter, attrgetter
import math
import random
import string
import timeit
from timeit import Timer as t
import matplotlib.pyplot as plt
import numpy as np

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

class mlp:
  def __init__(self, inputs, nhidden, targets):
    self.nin = inputs
    self.nhidden = nhidden
    self.nout = targets
    self.ai = [1.0]*self.nin
    self.ah = [1.0]*self.nhidden
    self.ao = [1.0]*self.nout
    self.wi = [ [0.0]*self.nhidden for i in range(self.nin) ]
    self.wo = [ [0.0]*self.nout for j in range(self.nhidden) ]
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )

  def mplRun (self, inputs):
    if len(inputs) != self.nin:
      print 'incorrect number of inputs'
    for i in range(self.nin):
      self.ai[i] = inputs[i]

    # print 'ai---',self.ai
    # print 'wi---',self.wi
    # print 'wi is:',np.shape(self.wi)
    for j in range(self.nhidden):
      self.ah[j] = sigmoid(sum([ self.ai[i]*self.wi[i][j] for i in range(self.nin) ]))
    for k in range(self.nout):
      self.ao[k] = sigmoid(sum([ self.ah[j]*self.wo[j][k] for j in range(self.nhidden) ]))
    # print 'ao---',self.ao
    return self.ao

  def weights(self):
    print 'Input weights:'
    for i in range(self.nin):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in range(self.nhidden):
      print self.wo[j]
    print ''

  def test(self, patterns):
    results, targets = [], []
    for p in patterns:
      inputs = p[0]
      rounded = [ round(i) for i in self.mplRun(inputs) ]
      if rounded == p[1]: result = '+++++'
      else: result = '-----'
      print '%s %s %s %s %s %s %s' %( 'Inputs:', p[0], '-->', str(self.mplRun(inputs)).rjust(65), 'Target', p[1], result)
      results+= self.mplRun(inputs)
      targets += p[1]
    return results, targets

  def sumErrors (self):
    error = 0.0
    for p in pat:
      inputs = p[0]  # Input data of each row
      targets = p[1] # Training targets value of each row, e.g. row 1 is class 0 flower.
      self.mplRun(inputs)
      error += self.calcError(targets)
    inverr = 1.0/error
    print 'return of sumErrors',inverr
    return inverr

  def calcError (self, targets):
    error = 0.0

    print 'calcError targets', targets

    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2

    print "targets and error",targets,'|',error
    return error

  def assignWeights (self, weights, I):
    io = 0
    for i in range(self.nin):
      for j in range(self.nhidden):
        self.wi[i][j] = weights[I][io][i][j]
    io = 1
    for j in range(self.nhidden):
      for k in range(self.nout):
        self.wo[j][k] = weights[I][io][j][k]

  def testWeights (self, weights, I):
    same = []
    io = 0
    for i in range(self.nin):
      for j in range(self.nhidden):
        if self.wi[i][j] != weights[I][io][i][j]:
          same.append(('I',i,j, round(self.wi[i][j],2),round(weights[I][io][i][j],2),round(self.wi[i][j] - weights[I][io][i][j],2)))

    io = 1
    for j in range(self.nhidden):
      for k in range(self.nout):
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
def makePops (pop):
  weights, errors = [], []
  for i in range(len(pop)):                 # for each individual
    weights.append([pop[i].wi,pop[i].wo])   # append input & output weights of individual to list of all pop weights
    errors.append(pop[i].sumErrors())       # append 1/sum(MSEs) of individual to list of pop errors
  fitnesses = calcFit(errors)              # fitnesses are a fraction of the total error

  # print "weight:" , weights
  # print 'errors: ',errors
  print 'fitness', fitnesses
  # for i in range(int(pop_size*0.15)):
  #   print str(i).zfill(2), '1/sum(MSEs)', str(errors[i]).rjust(15), str(int(errors[i]*graphical_error_scale)*'-').rjust(20), 'fitness'.rjust(12), str(fitnesses[i]).rjust(17), str(int(fitnesses[i]*1000)*'-').rjust(20)
  del pop
  print 'str pops: ',zip(weights, errors,fitnesses)
  return zip(weights, errors,fitnesses)            # weights become item[0] and fitnesses[1] in this way fitness is paired with its weight in a tuple

def rankPop (newpopW,pop):
  errors, copy = [], []           # a fresh pop of NN's are assigned to a list of len pop_size
  #pop = [NN(ni,nh,no)]*pop_size # this does not work as they are all copies of eachother

  pop= [mlp(ni,nh,no) for i in range(pop_size) ]
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

def iteratePop (rankedPop):
  rankedWeights = [ item[0] for item in rankedPop]
  print 'ranked weights',rankedWeights
  fitnessScores = [ item[-1] for item in rankedPop]
  newpopW = [ eval(repr(x)) for x in rankedWeights[:int(pop_size*0.15)] ]
  while len(newpopW) <= pop_size:
  # Breed two randomly selected but different chromos until pop_size reached
    ch1, ch2 = [], []
    index1 = roulette(fitnessScores)
    index2 = roulette(fitnessScores)
    while index1 == index2:
    # ensures different chromos are used for breeeding
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
max_iterations = 2
pop_size = 10
mutation_rate = 0.1
crossover_rate = 0.8
ni, nh, no = 4,6,1

def main ():
  # Rank first random population
  pop= [ mlp(ni,nh,no) for i in range(pop_size) ] # fresh pop
  # print len(pop)
  pops = makePops(pop)
  print "stringifyed pop: ",pops
  rankedPop = sorted(pops, key = itemgetter(-1), reverse = True) # THIS IS CORRECT

  # print "ranked pops: ", rankedPop
  # print rankedPop[0] in pops
  # Keep iterating new pops until max_iterations
  iters = 0
  tops, avgs = [], []
  while iters != max_iterations:
    # if iters%1 == 0:
      # print 'Iteration'.rjust(150), iters
    newpopW = iteratePop(rankedPop)
    print newpopW
    rankedPop, toperr, avgerr = rankPop(newpopW,pop)
    tops.append(toperr)
    avgs.append(avgerr)
    iters+=1

  # test a NN with the fittest weights
  tester = mlp (ni,nh,no)
  fittestWeights = [ x[0] for x in rankedPop ]
  tester.assignWeights(fittestWeights, 0)
  results, targets = tester.test(testpat)
  x = np.arange(0,150)
  """
  title2 = 'Test after '+str(iters)+' iterations'
  plt.title(title2)
  plt.ylabel('Node output')
  plt.xlabel('Instances')
  plt.plot( results, 'xr', linewidth = 0.5)
  plt.plot( targets, 's', color = 'black',linewidth = 3)
  #lines = plt.plot( results, 'sg')
  plt.annotate(s='Target Values', xy = (110, 0),color = 'black', family = 'sans-serif', size  ='small')
  plt.annotate(s='Test Values', xy = (110, 0.5),color = 'red', family = 'sans-serif', size  ='small', weight = 'bold')
  plt.figure(2)
  plt.subplot(121)
  plt.title('Top individual error evolution')
  plt.ylabel('Inverse error')
  plt.xlabel('Iterations')
  plt.plot( tops, '-g', linewidth = 1)
  plt.subplot(122)
  plt.plot( avgs, '-g', linewidth = 1)
  plt.title('Population average error evolution')
  plt.ylabel('Inverse error')
  plt.xlabel('Iterations')
  plt.show()

  """

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
    main()
