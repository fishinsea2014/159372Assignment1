__author__ = 'QSG'
import numpy as np
import pylab as pl

class SOM(object):

    # Initializing a weight matrix, the shape is D*(n* m), that is, n*m weight vector, each D dimension
    # X: N*D, the data set of input data
    #output: output layer is a two dimensions array
    #iteration: the times of iteration
    #batch_size: the quantity of samples for each iteration
    def __init__(self, X, output, iteration, batch_size):
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        print (self.W.shape)

    #Returns an integer, representing the topological distance,
    #the larger the time, the smaller the topological neighborhood.
    #'t' represent the times of iteration.
    def GetN(self, t):
        a = min(self.output)
        return int(a-float(a)*t/self.iteration)

    #Return a learning rate
    #'t' is the times of iteration.
    #'n' is the topological distance.
    def Geteta(self, t, n):
        return np.power(np.e, -n)/(t+2)


    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N+1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:,w], e*(X[x,:] - self.W[:,w]))

    #Return a list of neuron that need to be updated in different neighbourhood size.
    #'index' is the index of winner neuron
    #'N' is the neighbourhood size.
    def getneighbor(self, index, N):
        a, b = self.output
        length = a*b
        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N+1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    #Train the SOM, return an updated W
    #the shape of training sample is batch_size*(n*m)
    #winner is a one dimensional vector
    #batch_size is the index of the winning neuron

    def train(self):
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    #Return the winner of training
    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        print 'The winner are',winner, '\n'
        return winner

#Normalization input dataset
# return the result of normalization
#X is a two dimensions array, N*D
def normal_X(X):
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X

#Normalize W,D*(n*m), which is D of n*m data items.
def normal_W(W):
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:,i], W[:,i]))
        W[:, i] /= np.sqrt(temp)
    return W

#Draw the SOM result
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm','r','y','g','b','c','k']
    for i in range(len(C)):
        coo_X = []    #X-coordiante list
        coo_Y = []    #Y-coordiante list
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)

    pl.legend(loc='upper right')
    pl.show()

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
        # ln.append(type)
        # print d
        data.append(ln)
    type+=1
    f.close()

# Normalization data
data=np.array(data)
data[:,:43] = data[:,:43]-data[:,:43].mean(axis=0)
imax = np.concatenate((data.max(axis=0)*np.ones((1,43)),data.min(axis=0)*np.ones((1,43))),axis=0).max(axis=0)

# print "========================="
data[:,:43] = data[:,:43]/imax[:43]
data=np.mat(data)
data_org = data.copy()
# print 'Original dataset',data
for item in data_org:
    print item

#Start training
som = SOM(data, (3, 13), 5, 30)
som.train()
res = som.train_result()

classify = {}
for i, win in enumerate(res):
    if not classify.get(win[0]):
        classify.setdefault(win[0], [i])
    else:
        classify[win[0]].append(i)
# print output classes
for i in classify.keys():
    print 'Class-',i,':',classify.get(i),'\n'

#Draw the classify result.
classesD = []
for i in classify.values():
    classesD.append(data[i].tolist())
draw(classesD)