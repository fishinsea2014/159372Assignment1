# Stephen Marsland, 2008

# The four peaks fitness function
from numpy import *
def fourpeaks(population):

	T = 15
	start = zeros((shape(population)[0],1))
	finish = zeros((shape(population)[0],1))

	fitness = zeros((shape(population)[0],1))

	for i in range(shape(population)[0]):
		s = where(population[i,:]==1)
		f = where(population[i,:]==0)
		if size(s)>0:
			start = s[0][0]
		else:
			start = 0	
		
		if size(f)>0:
			finish = shape(population)[1] - f[-1][-1] -1
		else:
			finish = 0

		if start>T and finish>T:
			fitness[i] = maximum(start,finish)+100
		else:
			fitness[i] = maximum(start,finish)

	fitness = squeeze(fitness)
	return fitness
