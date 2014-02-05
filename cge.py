from cge_tools import *
import numpy as np
import pandas as pd


index = ['BRD', 'MLK', 'CAP', 'LAB', 'HH', 'GOV']

ii = ['BRD', 'MLK']
# i and j is ii

ih = ['CAP', 'LAB']
# h and k is ih

table = 
[
	['BRD', 'MLK'],
	['.....'],
	['.....'],
	['.....'],
	['.....'],
	['.....'],
	['.....']
]


sam = Sam(index)
print sam
#sam['HH']['BRD'] = 15
#sam['HH']['MLK'] = 35
#sam['BRD']['CAP'] = 5
#sam['MLK']['CAP'] = 20
#sam['BRD']['LAB'] = 10
#sam['MLK']['LAB'] = 15
#sam['CAP']['HH'] = 25
#sam['LAB']['HH'] = 25



for u in index:
	assert sum(sam[u]) == sum(sam.ix[u])

doc = {
	'X0': " household consumption of the i-th good ",
	'G0': " government consumption of i-th good or k-th factor ",
	'F0': " the h-th factor input by the j-th firm ",
	'Z0': " output of the j-th good"	
	# ...
}

X0 = sam.field_by_rows("HH", ii)

#G0 = sam.field_by_rows("gov", ii + ih)

F0 = sam.sub_matrix(rows=ih, columns=ii)

Z0 = sam.sum_by_rows(ii)

FF = sam.sum_by_rows(ih)

print sam
print doc['X0']
print X0.dtype.names
print X0
print 'F0'
print F0
print 'Z0'
print Z0.dtype.names
print Z0
print 'FF'
print FF.dtype.names
print FF


def alpha(i):
	""" share parameter in utility function """
	return X0(i) / sum([X0(j) for j in ii])


print 'alpha'
alpha = np.empty_like(X0)
sX0 = rsum(X0)
for i in ii:
	alpha[i] = X0[i] / sX0
print alpha.dtype.names
print alpha

print 'beta'
beta = F0.copy()
sF0 = F0.sum()

for h in ih:
	for j in ii:
		beta[j][h] = F0[j][h] / sF0[j]
print beta

print 'b'
b = Z0.copy()
for j in ii:
	b[j] = Z0[j] / cobbdouglas(F0, beta,  industry=j, factors=ih)

printseries(b)



UU = lambda x: np.prod(x ** alpha.view())