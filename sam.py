import numpy as np
from table_checks import check_table, check_balance
import table
from balance import balance

class Sam(table.Table):
    def __init__(self, index, sam=None, autobalance=False, unbalanced=False):
        check_table(sam, len(index))
        if autobalance:
            if not(check_balance(sam)):
                sam = balance(sam)
        else if not(unbalanced):
            assert check_balance(sam), 'sam not balanced \nif you want to assign an unbalance table, use either the autobalance parmeter or unbalanced'
        super(Sam, self).__init__(index, index, sam)

    @classmethod
    def empty(cls, index):
        sam = np.zeros(shape=(len(index), len(index)))
        return cls(index=index, sam=sam)

    def balance(self):
        self.sam = balance(self.sam)