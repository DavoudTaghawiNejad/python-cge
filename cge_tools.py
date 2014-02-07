import numpy as np
import pandas as pd


class Series:
	def __init__(self, names, data, doc=''):
		self.names = names
		self.data = data
 	
 	@classmethod
	def like(Cls, like, doc=''):
		return Cls(names=like.names, data=np.empty_like(like.data), doc=doc)
	
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
		self.data[i] = value

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


class Sam:

	def __init__(self, index, columns, table=None):
		if table ==  None:
			table = np.zeros(shape=(len(index), len(index)))
		self.sam = pd.DataFrame(table,index=index,columns=columns)

	@classmethod
	def unflatten(Cls, index, columns, table):
		return Cls(table=np.reshape(table, newshape=(len(index), len(index))), index=index, columns=columns)

	def __getitem__(self, row):
		return self.sam[row]

	def __repr__(self):
		return self.sam.__repr__()

	def inputs(self, inputs, to):
		fields = empty_recarray(inputs)
		for i, I in enumerate(inputs):
			fields[i] = self.sam[to][I]
		return Series(inputs, fields)

	def sub_matrix(self, columns, rows):
		matrix = pd.DataFrame(index=rows, columns=columns)
		for c in columns:
			for r in rows:
				matrix[c][r] = self.sam[c][r]
		return matrix

	def sum_by_rows(self, rows):
		sums = empty_recarray(rows)
		for r, R in enumerate(rows):
			sums[r] = sum(self.sam.ix[R])
		return Series(rows, sums)

	def sum_selected_columns_by_rows(self, rows, columns):
		sums = empty_recarray(rows)
		for r, R in enumerate(rows):
			sums[r] = sum([self.sam.ix[R][C] for C in columns])
		return Series(rows, sums)
