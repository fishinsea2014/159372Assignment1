# The pollen classification by MLP

# Extract data from .dat files.
import os
filelist=os.listdir(".\\Pollens")
data=[]
type=0
for files in filelist:
    f=open('.\\Pollens\\'+files,'r')
    for lines in f.readlines():
        l=[]
        l.extend(lines.strip().split(" "))
        # print l
        ln=[]
        for n in l:
            if n!="":
                ln.append(float(n.strip()))
        ln.append(type)
        # print d
        data.append(ln)
    type+=1
    f.close()
# The data set has 650 lines, 13 types of pollen, each pollen has 50 lines data, and each item has 43 numbers.

#Divide data into three test sets, which are: training set, test set and validate set
from numpy import *
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
print target

# Randomly order the data
ordr = range(shape(data)[0])
random.shuffle(ordr)
data = data[ordr,:]
target = target[ordr,:]

train = data[::5,0:43]
print "Length of train data set",len(train)
traint = target[::5]
valid = data[1::6,0:43]
validt = target[1::6]
test = data[3::4,0:43]
testt = target[3::4]

print train
# Train the network
import mlp
# hiddenNodes=range(5,30,3)
hiddenNodes=[130,261,390]
# hiddenNodes=[5,10,15]
correctRate=[]
for i in hiddenNodes:
    net = mlp.mlp(train,traint,i,outtype='softmax')
    net.earlystopping(train,traint,valid,validt,0.1)
    correctRate.append(net.confmat(test,testt))

print '===Test result==='
print hiddenNodes
print correctRate
print '=================='

for i in range(len(hiddenNodes)):
    print 'Hidden nodes: ',hiddenNodes[i], '|| Correct percentage: ',correctRate[i],'\n'