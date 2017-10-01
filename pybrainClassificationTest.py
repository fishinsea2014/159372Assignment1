__author__ = 'QSG'
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from pylab import ion,ioff,figure, draw, contourf,clf,show, hold,plot
from scipy import diag,arange,meshgrid,where
from numpy.random import multivariate_normal

#use a 2D dataset, classify into 3 classes
means=[(-1,2),(2,4),(3,1)]
cov=[diag([1,1]),diag([0.5,1.2]),diag([1.5,0.7])]
alldata=ClassificationDataSet(2,1,nb_classes=3)
for n in xrange(400):
    for kclass in range(3):
        input  = multivariate_normal(means[kclass],cov[kclass])
        # print 'input: ', input
        alldata.addSample(input,[kclass])
# print alldata

tstdata,trndata=alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "Number of training patterns: ", len(trndata)
print "input and output dimensions: ",trndata.indim,',',trndata.outdim
print "first sample (input, target,class):"
# print trndata['input'][0],trndata['target'][0],trndata['class'][0]
# print trndata['input'],trndata['target'],trndata['class']
# print len(trndata['target'])
# print trndata['target']
# print trndata['class']
print trndata.outdim

#build a 2,5,3 network
fnn=buildNetwork(trndata.indim,5,trndata.outdim,outclass=SoftmaxLayer)

#train the network
trainer=BackpropTrainer(fnn,dataset=trndata,momentum=0.1,verbose=True,weightdecay=0.01)

#make a grid data pot
ticks=arange(-3,6,0.2)
X,Y=meshgrid(ticks,ticks)
griddata=ClassificationDataSet(2,1,nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]],[0])
griddata._convertToOneOfMany()

for i in range(20):
    trainer.trainEpochs(1)
    trnresult=percentError(trainer.testOnClassData(),trndata['class'])
    tstresult=percentError(trainer.testOnClassData(dataset=tstdata),tstdata['class'])
    print 'epoch: %4d' % trainer.totalepochs,\
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
    out=fnn.activateOnDataset(griddata)
    out=out.argmax(axis=1)
    out=out.reshape(X.shape)

    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    # hold(True) # overplot on
    for c in [0,1,2]:
		here, _ = where(tstdata['class']==c)
    plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
		contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot
    ioff()
    show()


