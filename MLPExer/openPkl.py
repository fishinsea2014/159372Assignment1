__author__ = 'QSG'
import cPickle as pickle
f = open('mnist.pkl')

info = pickle.load(f)
print info   #show file