'''
Created on Oct 7, 2012

@author: srmarsla
'''

import pylab as pl
import numpy as np
import pcn
import cPickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz','rb')
tset, vset, teset = cPickle.load(f)
f.close()

nread = 200
# Just use the first few images
train_in = tset[0][:nread,:]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((nread,10))
for i in range(nread):
    train_tgt[i,tset[1][i]] = 1

test_in = teset[0][:nread,:]
test_tgt = np.zeros((nread,10))
for i in range(nread):
    test_tgt[i,teset[1][i]] = 1

# Train a Perceptron on training set
p = pcn.pcn(train_in, train_tgt)
p.pcntrain(train_in, train_tgt,0.25,100)

# This isn't really good practice since it's on the training data, 
# but it does show that it is learning.
p.confmat(train_in,train_tgt)

# Now test it
p.confmat(test_in,test_tgt)

#-----
# Now for an MLP
print "-----"
import mlp

# We will need the validation set
valid_in = vset[0][:nread,:]
for i in range(nread):
    valid_tgt[i,vset[1][i]] = 1

net = mlp.mlp(train,traint,5,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)

net = mlp.mlp(train_in,train_tgt,5,outtype='softmax')
net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1)
net.confmat(test_in,test_tgt)