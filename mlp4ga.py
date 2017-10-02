import numpy as np
import math
import random

class mlp:
  def __init__(self, inputs, nhidden, targets):
    self.nin = np.shape(inputs)[1] #colomns
    self.nout = np.shape(targets)[1]
    self.ndata = np.shape(inputs)[0] #rows
    self.nhidden=nhidden
    self.targets=targets
    self.ai = inputs
    self.ah = [1.0]*self.nhidden
    self.ao = [1.0]*self.nout
    # self.wi = [ [0.0]*self.nhidden for i in range(self.nin) ]
    # self.wo = [ [0.0]*self.nout for j in range(self.nhidden) ]
    # randomizeMatrix ( self.wi, -0.2, 0.2 )
    # randomizeMatrix ( self.wo, -2.0, 2.0 )

    # Initialise network
    self.wi = (np.random.rand(self.nin,self.nhidden)-0.5)*2/np.sqrt(self.nin)
    self.wo = (np.random.rand(self.nhidden,self.nout)-0.5)*2/np.sqrt(self.nhidden)


  def mplRun (self, inputs):
      """ Run the network forward """
      ai=inputs
      # print 'ai---',ai
      # print 'wi---',self.wi
      # print 'wi is:',np.shape(self.wi)

      # for j in range(self.nhidden):
      #     for i in range (self.nin):
      #         print 'input:', input
      #         print 'wi i,j:',self.wi[i][j]
      #         input[i]*self.wi[i][j]

      for j in range(self.nhidden):
        self.ah[j] = math.tanh(sum([ inputs[i]*self.wi[i][j] for i in range(self.nin) ]))

      # print 'ah---',self.ah
      # print 'nout---',self.nout
      # print 'wo---',self.wo
      for k in range(self.nout):
          tmp=math.tanh(sum([ self.ah[j]*self.wo[j][k] for j in range(self.nhidden) ]))
          # print 'temp---',tmp
          self.ao[k] = tmp

      self.ao=self.ao[:1]
      # print 'self.ao---',self.ao[:1]

      return self.ao[:1]

  # def weights(self):
  #   # print 'Input weights:'
  #   for i in range(self.nin):
  #     print self.wi[i]
  #   print 'Output weights:'
  #   for j in range(self.nhidden):
  #     print self.wo[j]
  #   print ''

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

    for i in range(len(self.ai)):
        self.mplRun(self.ai[i])
        error += self.calcError(self.targets[i])

    # for p in self.ai:
    #   inputs = p[0]
    #   targets = p[1]
    #   self.mplRun(p)
    #   error += self.calcError(self.ao)
    inverr = 1.0/error
    return inverr

  def calcError (self,target):
    error = 0.0
    # print 'calcError targets',targets
    for k in range(len(target)):
      error += 0.5 * (target[k]-self.ao[k])**2
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

