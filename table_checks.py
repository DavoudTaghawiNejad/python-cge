import numpy as np


def check_table(table, length):
    try:
        assert len(table) == length, (
                            'index is %i fields long, but %i rows in table' 
                            % (length, len(table))
        )
        for r, row in enumerate(table):
            assert len(row) == length, (
                    'index is %i fields long, but row %i is only %i fields long' 
                            % (length, i, len(row))
            )
    except TypeError, e:
        assert table.shape[0] == table.shape[1]  == length

def check_square(table):
    try:
        length = len(table)
        for r, row in enumerate(table):
            assert len(row) == length, (
                    'index is %i fields long, but row %i is only %i fields long' 
                            % (length, i, len(row))
            )
    except TypeError, e:
        assert table.shape[0] == table.shape[1]  == length

def check_balance(table):
    assert (
                    (np.sum(table, 1) - 1e-06 < np.sum(table, 0)).all()
                and (np.sum(table, 1) + 1e-06 > np.sum(table, 0)).all()
    ), np.sum(table, 0) - np.sum(table, 1)

