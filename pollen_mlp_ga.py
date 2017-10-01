__author__ = 'QSG'
from numpy import *
from neurolab import *
# import readPollenData
import mlp4ga


#Divide data into three test sets, which are: training set, test set and validate set
data=array(data)

# Normalization data
data[:,:43] = data[:,:43]-data[:,:43].mean(axis=0)
imax = concatenate((data.max(axis=0)*ones((1,44)),data.min(axis=0)*ones((1,44))),axis=0).max(axis=0)
# print "========================="
# print imax
data[:,:43] = data[:,:43]/imax[:43]
# print "====================="
# print data[0:44,:]
#
# # Split into training, validation, and test sets
target = zeros((shape(data)[0],13))
for i in range(13):
    indices = where(data[:,43]==i)
    # print indices
    target[indices,i] = 1
# print target

# Randomly order the data
ordr = range(shape(data)[0])
random.shuffle(ordr)
data = data[ordr,:]
target = target[ordr,:]

#Convert target to neurolab format
neutarget=[]
for i in target:
    for j in range(len(i)):
        if i[j]==1:
            neutarget.append([j+1])
neutarget=array(neutarget)
# print neutarget


train = data[::5,0:43]
train=array(train)
print "Length of train data set",len(train)
traint = neutarget[::5]
print traint
valid = data[1::6,0:43]
validt = target[1::6]
test = data[3::4,0:43]
testt = target[3::4]

print train
print len(traint)

#Building neuro network.
inputnum=130 #Number of input nodes
hiddennum=261 #Number of hidden nodes
outputnum=1  #NUmber of output nodes
# inputneuro=[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]
inputneuro=[[-1,1] for i in range(43)]
print inputneuro
net_pollen=net.newff(inputneuro,[hiddennum,outputnum])
print len(train[0])
print traint
print traint.shape[1]
err=net_pollen.train(train,traint)
err=net_pollen.sim()
print err

#Setup GA constants.
maxgen=50 #Iteration times
sizepop=100 #Population size
pmutation=0.01 #mutation rate
pcross=0.7 #Crossover rate

#Total number of nodes
numsum=inputnum*hiddennum + hiddennum+hiddennum*outputnum+outputnum
print numsum
lenchrom=ones((1,numsum))
print lenchrom

# bound=[-3*ones((numsum,1))3*ones(numsum,1)]

