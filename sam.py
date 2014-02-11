import numpy as np
import pandas as pd
from cge_tools import Series, empty_recarray
from table_checks import check_table, check_balance
import table

class Sam(table.Table):
    def __init__(self, index, sam=None):
        check_table(sam, len(index))
        check_balance(sam)
    	super(Sam, self).__init__(index, index, sam)

    @classmethod
    def empty(cls, index):
        sam = np.zeros(shape=(len(index), len(index)))
        return cls(index=index, sam=sam)