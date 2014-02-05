import numpy as np
import pandas as pd

def names(data):
	return data.dtype.names

def rsum(recarray):
	return np.float64(recarray.view().sum(), dtype=np.float64)

def cobbdouglas(factor_table, beta, industry, factors):
	return np.prod([factor_table[industry][f] ** beta[industry][f] for f in factors])
	 

def printseries(data, name=''):
	print name
	print data.dtype.names
	print data

class Sam:

	def __init__(self, index, table=None):
		if table ==  None:
			table = np.zeros(shape=(len(index), len(index)))
		self.sam = pd.DataFrame(table,index=index,columns=index)
		self.ix = self.sam.ix

	def __getitem__(self, row):
		return self.sam[row]

	def __repr__(self):
		return self.sam.__repr__()

	def field_by_rows(self, column, rows):
		fields = np.recarray(1, names=(rows), formats=['f8',]  * len(rows))
		for i in rows:
			fields[i] = self.sam[column][i]
		return fields

	def sub_matrix(self, columns, rows):
		matrix = pd.DataFrame(index=rows, columns=columns)
		for c in columns:
			for r in rows:
				matrix[c][r] = self.sam[c][r]
		return matrix

	def sum_by_rows(self, rows):
		sums = np.recarray(1, names=(rows), formats=['f8',]  * len(rows))
		for r in rows:
			sums[r] = sum(self.sam.ix[r])
		return sums		

	def sum_selected_columns_by_rows(self, rows, columns):
		sums = np.recarray(1, names=(rows), formats=['f8',]  * len(rows))
		for r in rows:
			sums[r] = sum([self.sam.ix[r][c] for c in columns])
		return sums
