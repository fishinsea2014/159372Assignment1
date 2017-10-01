__author__ = 'QSG'
import pickle,gzip,numpy
import pylab as pl

f=gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set=pickle.load(f,encoding='iso-8859-1')
f.close()
pl.imshow(numpy.reshape(train_set[0][0,:],[28,28]))
print (train_set,valid_set,test_set)

