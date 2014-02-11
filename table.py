import numpy as np
import pandas as pd
from cge_tools import Series, empty_recarray


class Table(object):
    def __init__(self, index, columns, sam):
        self.sam = pd.DataFrame(sam, index=index, columns=columns)
        self.index=self.sam.index
        self.columns=self.sam.columns

    @classmethod
    def empty(cls, index, columns):
        sam = np.zeros(shape=(len(index), len(columns)))
        return cls(index=index, columns=columns, sam=sam)

    @classmethod
    def unflatten(cls, index, columns, sam):
        return cls(sam=np.reshape(
                                    sam, 
                                    newshape=(len(index), len(columns))), 
                                    index=index, 
                                    columns=columns
                        )

    def __getitem__(self, row):
        return self.sam[row]
    
    def __setitem__(self, row, value):
        self.sam[row] = value

    def __repr__(self):
        return self.sam.__repr__()

    def __str__(self):
        return str(self.sam)

    def __len__(self):
        return len(self.sam)

    def __delitem__(self, row):
        del self.sam[row]

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

    def matrix(self):
        return np.matrix(self.sam)

    def replace(self, m):
        return Table(sam=self.sam, index=self.sam.index, columns=self.sam.columns)

    def array(self):
        return self.sam.values