import numpy as np
from openopt import NLP


def balance(table, debug=False):
    table = np.matrix(table)
    assert table.shape[0] == table.shape[1]
    size = table.shape[0]

    def objective(ox):
        ox = ox.reshape(size, size)
        ox = np.square((ox - table) / ox)
        return np.sum(ox)

    def constraints(ox):
        ox = ox.reshape(size, size)
        ret = np.sum(ox, 0) - np.sum(ox, 1)
        return ret

    if debug:
        print("--- balance ---")
    p = NLP(objective, table, h=constraints, iprint = 50 * int(debug), maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1') 
    r = p.solve('ralg', plot=0)
    if debug:        
        print dir(r)
        print constraints(r.xf)
    return r.xf.reshape(size, size)