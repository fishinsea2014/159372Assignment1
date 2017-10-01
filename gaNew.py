__author__ = 'QSG'
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.optimization.populationbased.ga import GA
from pybrain.tools.shortcuts import buildNetwork

d=ClassificationDataSet(3)
d.addSample([0,0,0],[0.])
d.addSample([0,1,0],[1.])
d.addSample([1,0,0],[1.])
d.addSample([1,1,0],[0.])
d.setField('class',[[0.],[1.],[1.],[0.]])

nn=buildNetwork(3,3,1)

print nn.activate([0,1,1])
ga=GA(d.evaluateModuleMSE,nn, minimize=True)
for i in range(100):
    nn=ga.learn(0)[0]

print nn.activate([0,1,1])[0]

# print nn
