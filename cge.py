import numpy as np
import pandas as pd
u = ['BRD', 'MLK', 'CAP', 'LAB', 'HH', 'GOV']

ii = ['BRD', 'MLK']

# i is j

ih = ['CAP', 'LAB']

# h is k

table = [
		['.....'],
		['.....'],
		['.....'],
		['.....'],
		['.....'],
		['.....'],
		['.....']
	]

table = np.random.randint(0,9, size=(7,7))


sam = pd.DataFrame(table,index=u,columns=u)


def X0(i):
	""" household consumption of the i-th good """
	assert ii.count(i), i
	return sam["HH"][i]

def G0(g):
	""" government consumption of i-th good or k-th factor """
	assert ii.count(g) or ih.count(g), g
	return sam[i]["gov"]	

def F0(h, j):
	""" ... """
	assert ih.count(h) and ii.count(j)
	return sam[h][j]

def Z0(j):
	""" ... """
	assert ii.count(j), j
	return sum(sam.ix[j])

def FF(h):
	""" ... """
	assert ih.count(h), h
	return sum(sam[h])

print sam
print X0.__doc__
print [X0(i) for i in ii]
# ...

def alpha(i):
	""" share parameter in utility function """
	return X0(i) / sum([X0(j) for j in ii])

def beta(h, j):
	
