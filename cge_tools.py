import numpy as np
import pandas as pd


class Series:
	def __init__(self, names, data, doc=''):
		self.names = names
		self.data = data
		self.doc = doc
 	
 	@classmethod
	def like(Cls, like, doc=''):
		return Cls(names=like.names, data=np.empty_like(like.data), doc=doc)

	@classmethod 
	def empty(Cls, names, doc=''):
		return Cls(names=names, data=np.empty((len(names)), dtype='f64'), doc=doc)

	def __repr__(self):
		return self.names.__repr__() + '\n' + self.data.__str__()

	def __get__(self):
		return self.data

	def __iter__(self):
		return self.data.__iter__()

	def __getitem__(self, i):
		try:
			return self.data[i]
		except ValueError:
			return self.data[self.names.index(i)]

	def __setitem__(self, i, value):
		try:
			self.data[i] = value
		except ValueError:
			self.data[self.names.index(i)] = value

	def __rpow__(self, x):
		return x ** self.data

	def __len__(self):
		return len(self.data)


def names(data):
	return data.dtype.names

def rsum(recarray):
	return np.float64(sum(recarray), dtype=np.float64)

def cobbdouglas(factor_table, beta, industry, factors):
	return np.prod([factor_table[industry][f] ** beta[industry][f] for f in factors])

def empty_recarray(names):
	return np.empty(len(names), dtype='f64')


