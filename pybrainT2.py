__author__ = 'QSG'
from pybrain.structure import FeedForwardNetwork

#Build a forward network
n=FeedForwardNetwork()
from pybrain.structure import LinearLayer, SigmoidLayer
#Create input, hidden, output layer
inLayer=LinearLayer(2, name="inLayer")
hiddenLayer=SigmoidLayer(2)
outLayer=LinearLayer(1, name='outLayer')

#Add layers into network
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

#add more input and output modele, setup connections
from pybrain.structure import FullConnection
in_to_hidden=FullConnection(inLayer,hiddenLayer)
hidden_to_out=FullConnection(hiddenLayer,outLayer)

#Add the connections to the network
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

#Make the MLP usable by sortModules method
n.sortModules()

#Active the neuroNetwork
activeNetwork=n.activate([1,2])
print activeNetwork

#check the parameters
print in_to_hidden.params
print n.params
#Check the network structure
print n

#name the network





