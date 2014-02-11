import numpy as np
from openopt import NLP


np.set_printoptions(suppress=True)

def balance(table, debug=False):
    table = np.array(table)
    assert table.shape[0] == table.shape[1]
    size = table.shape[0]
    x0 = np.array([v for v in table.flatten() if v !=0])

    def transform(ox):
        ret = np.zeros_like(table)
        i = 0
        for r in range(size):
            for c in range(size):
                if table[r, c] != 0:
                    ret[r, c] = ox[i]
                    i += 1
        return ret
    
    def objective(ox):
        ox = np.square((ox - x0) / x0)
        return np.sum(ox)

    def constraints(ox):
        ox = transform(ox)
        ret = np.sum(ox, 0) - np.sum(ox, 1)
        return ret

    print constraints(x0)

    if debug:
        print("--- balance ---")
    p = NLP(objective, x0, h=constraints, iprint = 50 * int(debug), maxIter = 100000, maxFunEvals = 1e7, name = 'NLP_1') 
    r = p.solve('ralg', plot=0)
    if debug:        
        print 'constraints'
        print constraints(r.xf)
    assert r.isFeasible

    return transform(r.xf)