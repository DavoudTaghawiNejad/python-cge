
""" Textbook of Computable General Equilibrium Modeling Page 72(93) 
	output 81 (102)
"""
from cge_tools import *
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from openopt import NLP
from pprint import pprint


np.set_printoptions(precision=3)

def eqX(industry, Industry):  ###
	i, I = industry, Industry
	def equation(x):
		X, px, pf = x[sX:sF], x[spx:spz], x[spf:epf]
		pf = np.array([float(x[spf:epf]), 1])
		alpha[I]
		[pf[h] * FF[H] for h, H in enumerate(ih)]
		px[i]
		return X[i] - alpha[I] * sum([pf[h] * FF[H] for h, H in enumerate(ih)] / px[i])
	return equation

def eqpx(i, I):
	def equation(x):
		X, Z = x[sX:sF], x[sZ:spx]
		return X[i] - Z[i]
	return equation

def eqZ(i, I): ###
	def equation(x):
		px, pz = x[spx:spz], x[spz:spf]
		return px[i] - pz[i]
	return equation

def eqpz(j, J):  ###
	def equation(x):
		F, Z = Sam.unflatten(index=ih, columns=ii, table=x[sF:sZ]), x[sZ:spx]
		return Z[j] - b[J] * np.prod([F[J][H] ** beta[J][H] for H in ih])
	return equation

def eqpf(h, H):
	def equation(x):
		F = Sam.unflatten(index=ih, columns=ii, table=x[sF:sZ])
		return sum(F[J][H] for J in ii) - FF[H]
	return equation

def eqF(h, H, j, J):  ###
	def equation(x):
		F, Z, pz, pf = Sam.unflatten(index=ih, columns=ii, table=x[sF:sZ]), x[sZ:spx], x[spz:spf], x[spf:epf]
		pf = np.array([float(x[spf:epf]), 1])
		return F[J][H] - beta[J][H] * pz[j] * Z[j] / pf[h]
	return equation

index = ['BRD', 'MLK', 'CAP', 'LAB', 'HH', 'GOV']

ii = ['BRD', 'MLK']
ri = xrange(len(ii))
# i and j is ii

ih = ['CAP', 'LAB']
# h and k is ih

table = [
			['BRD', 'MLK'],
			['.....'],
			['.....'],
			['.....'],
			['.....'],
			['.....'],
			['.....']
		]


sam = Sam(index=index, columns=index)

sam['HH']['BRD'] = 15
sam['HH']['MLK'] = 35
sam['BRD']['CAP'] = 5
sam['MLK']['CAP'] = 20
sam['BRD']['LAB'] = 10
sam['MLK']['LAB'] = 15
sam['CAP']['HH'] = 25
sam['LAB']['HH'] = 25


doc = {
	'X0': "X0 household consumption of the i-th good ",
	'G0': "G0 government consumption of i-th good or k-th factor ",
	'F0': "F0 the h-th factor input by the j-th firm ",
	'Z0': "Z0 output of the j-th good",
	'alpha': "alpha share parameter in utility function"
	# ...
}

X0 = sam.inputs(ii, to="HH")

# G0 = sam.field_by_rows("gov", ii + ih)

F0 = sam.sub_matrix(rows=ih, columns=ii)

Z0 = sam.sum_by_rows(ii)

FF = sam.sum_by_rows(ih)

print sam
print doc['X0']
print X0
print 'F0'
print F0
print 'Z0'

print Z0
print 'FF'

print FF
print 'alpha'
alpha = Series.like(X0)
sX0 = rsum(X0)
for i in ri:
	alpha[i] = X0[i] / sX0
print alpha

print 'beta'
beta = F0.copy()
sF0 = F0.sum()

for h in ih:
	for j in ii:
		beta[j][h] = F0[j][h] / sF0[j]
print beta

print 'b'
b = Series.like(Z0)
for j, J in enumerate(ii):

	b[j] = Z0[j] / cobbdouglas(F0, beta,  industry=J, factors=ih)

print(b)

equations_doc = {
	'UU(x)': 'utility / target',
	'eqX(i)': 'household demand function'
}

px = empty_recarray(ii)

j = i = len(ii)
h = len(ih)
sX = 0
sF = sX + i
sZ = sF + i * h
spx = sZ + j
spz = spx + i
spf = spz + j
epf = spf + h - 1


bnds = [(np.float64(0.001), 999999999)] * epf
lb = np.array([(np.float64(0.001))] * epf)
ub = np.array([(None)] * epf)
x = np.empty(epf, dtype='f64')
x[sX:sF] = X0.data
x[sF:sZ] = F0.as_matrix().flatten()
x[sZ:spx] = Z0.data
x[spx:spz] = [1] * i
x[spz:spf] = [1] * j
x[spf:epf] = [1] * (epf - spf)

t = x
x = np.array([21.1] * epf)
print x

xnames = [] * epf
xnames[sX:sF] = X0.names
xnames[sF:sZ] = [i+h+' ' for i in ii for h in ih]
xnames[sZ:spx] = Z0.names
xnames[spx:spz] = ii
xnames[spz:spf] = ii
xnames[spf:epf] = ih

xtypes = [] * epf
xtypes[sX:sF] = ['X0'] * len(X0)
xtypes[sF:sZ] = ['F'] * (len(ii) + len(ih))
xtypes[sZ:spx] = ['F0'] * len(F0)
xtypes[spx:spz] = ['pb'] * len(ii)
xtypes[spz:spf] = ['pz'] * len(ii)
xtypes[spf:epf] = ['pf'] * len(ih)

xnametypes = zip(xtypes,xnames)
pprint(zip(xnametypes, x))

constraints = []

qX = []
qpx = []
qZ = []
qpz = []
qpf = []
qF = []

def unlink(value):
	return value;

for i, I in enumerate(ii):	
	industry = unlink(i)
	Industry = unlink(I)
	
	constraints.append(eqX(industry, Industry)) # 1
	constraints.append(eqpx(industry, Industry))
	constraints.append(eqZ(industry, Industry))
	constraints.append(eqpz(industry, Industry))
	
del i
	

for f, F in enumerate(ih):
	factor = unlink(f)
	Factor = unlink(F)
	constraints.append(eqpf(factor, Factor))

for f, F in enumerate(ih):
	for i, I in enumerate(ii):
		factor = unlink(f)
		Factor = unlink(F)			
		industry = unlink(i)
		Industry = unlink(I)
		constraints.append(eqF(factor, Factor, industry, Industry))


UU = lambda x: - np.prod([x[i] ** alpha[i] for i in range(len(ii))])



p = NLP(UU, x, h=constraints, ub=ub, lb=lb, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')
#res = minimize(UU, x, method='SLSQP', bounds=bnds, constraints=constraints)
p.plot = True

solver = 'ralg'
r = p.solve(solver, plot=0) # string argument is solver name
print r.ff, '(', UU(t), UU(x), ')'
pprint(zip(xnametypes,zip(t,r.xf)))

for i, constraint in enumerate(constraints):
	print(i, '%02f' % constraint(r.xf))