__author__ = 'QSG'
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
input = np.array([[4,11],[7,340],[10,95],[3,29],[7,43],[5,128],[6,89],[9,789]])
target=np.array([[1],[0],[2],[0],[1],[2],[3],[0]])
net=nl.net.newff([[3,10],[11,340]],[5,1])
err=net.train(input,target,epochs=500, show=1, goal=0.02)
out=net.sim(input)
mymean=np.mean(out)
x_max=np.max(input[:,0])+5
x_min=np.min(input[:,0])-5
y_max=np.max(input[:,1])+5
y_min=np.min(input[:,1])-5
plt.subplot(211)
plt.plot(range(len(err)),err)
plt.xlabel('Epoch number')
plt.ylabel('err (default SSE)')
plt.subplot(212)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
for i in xrange(0,len(input)):
    if out[i]>mymean:
        plt.plot(input[i,0],input[i,1],'ro')
    else:
        plt.plot(input[i,0],input[i,1],'r*')

plt.show()