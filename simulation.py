from copy import copy
from cge_tools import empty_recarray
import numpy as np
from openopt import NLP
from sam import Sam


def unlink2(value, Value):
    return value, Value;

def unlink(value):
    return value;

class Simulation:
    """ equations_doc = {
    'UU(x)': 'utility / target',
    'eqX(i)': 'household demand function'
    """
    def __init__(self, calibration, parameter, debug=False):
        def eqX(industry, Industry):      
            i, I = industry, Industry
            def equation(x):
                X, px, pf = x[sX:sF], x[spx:spz], x[spf:epf]
                pf = np.array([float(x[spf:epf]), 1])
                return X[i] - self.calibration.alpha[I] * sum([pf[h] * self.calibration.FF[H] for h, H in enumerate(parameter.factors)] / px[i])
            return equation

        def eqpx(i):
            def equation(x):
                X, Z = x[sX:sF], x[sZ:spx]
                return X[i] - Z[i]
            return equation

        def eqZ(i): 
            def equation(x):
                px, pz = x[spx:spz], x[spz:spf]
                return px[i] - pz[i]
            return equation

        def eqpz(j, J):      
            def equation(x):
                F, Z = Sam.unflatten(index=parameter.factors, columns=parameter.industries, table=x[sF:sZ]), x[sZ:spx]
                return Z[j] - self.calibration.b[J] * np.prod([F[J][H] ** self.calibration.beta[J][H] for H in parameter.factors])
            return equation

        def eqpf(H):
            def equation(x):
                F = Sam.unflatten(index=parameter.factors, columns=parameter.industries, table=x[sF:sZ])
                return sum(F[J][H] for J in parameter.industries) - self.calibration.FF[H]
            return equation

        def eqF(h, H, j, J):      
            def equation(x):
                F, Z, pz, pf = Sam.unflatten(index=parameter.factors, columns=parameter.industries, table=x[sF:sZ]), x[sZ:spx], x[spz:spf], x[spf:epf]
                pf = np.array([float(x[spf:epf]), 1])
                return F[J][H] - self.calibration.beta[J][H] * pz[j] * Z[j] / pf[h]
            return equation
        self.calibration = copy(calibration)        
        self.calibration.X0 = self.calibration.X0['HH']
        j = i = len(parameter.industries)
        h = len(parameter.factors)
        sX = 0
        sF = sX + i
        sZ = sF + i * h
        spx = sZ + j
        spz = spx + i
        spf = spz + j
        epf = spf + h - 1
        epf_numerair = epf + 1
        lb = np.array([(np.float64(0.001))] * epf)
        self.x = x = np.empty(epf, dtype='f64')
        x[sX:sF] = self.calibration.X0.data
        x[sF:sZ] = self.calibration.F0.as_matrix().flatten()
        x[sZ:spx] = self.calibration.Z0.data
        x[spx:spz] = [1] * i
        x[spz:spf] = [1] * j
        x[spf:epf] = [1] * (epf - spf)       
        self.t = x[:]
        if debug:
            self.x = x = np.array([21.1] * epf)
        print x
        xnames = [] * (epf_numerair)
        xnames[sX:sF] = self.calibration.X0.names
        xnames[sF:sZ] = [i+h+' ' for i in parameter.industries for h in parameter.factors]
        xnames[sZ:spx] = self.calibration.Z0.names
        xnames[spx:spz] = parameter.industries
        xnames[spz:spf] = parameter.industries
        xnames[spf:epf] = parameter.factors
        xnames[epf] = parameter.factors[-1]

        xtypes = [] * epf_numerair
        xtypes[sX:sF] = ['X0'] * len(self.calibration.X0)
        xtypes[sF:sZ] = ['F'] * (len(parameter.industries) + len(parameter.factors))
        xtypes[sZ:spx] = ['F0'] * len(self.calibration.F0)
        xtypes[spx:spz] = ['pb'] * len(parameter.industries)
        xtypes[spz:spf] = ['pz'] * len(parameter.industries)
        xtypes[spf:epf] = ['pf'] * len(parameter.factors)
        xtypes[epf] = ['pf']

        self.xnametypes = ['%s %s' % (xnames[i], xtypes[i]) for i in range(epf - 1)]

        constraints = []
        for i, I in enumerate(parameter.industries):    
            industry, Industry = unlink2(i, I)
            constraints.append(eqX(industry, Industry)) 
            constraints.append(eqpx(industry))
            constraints.append(eqZ(industry))
            constraints.append(eqpz(industry, Industry))
                    
            

        for F in parameter.factors:
            Factor = unlink(F)
            constraints.append(eqpf(Factor))

        for f, F in enumerate(parameter.factors):
            for i, I in enumerate(parameter.industries):
                industry, Industry = unlink2(i, I)
                factor, Factor = unlink2(f, F)
                constraints.append(eqF(factor, Factor, industry, Industry))


        self.UU = UU = lambda x: - np.prod([x[i] ** self.calibration.alpha[i] for i in range(len(parameter.industries))])



        p = NLP(UU, x, h=constraints, lb=lb, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')        
        p.plot = debug
        self.r = p.solve('ralg', plot=0)

        if debug: 
            for i, constraint in enumerate(constraints):
                print(i, '%02f' % constraint(self.r.xf))
            

    def __str__(self):
        r = self.r
        ret = '%02f (%02f, %02f)\n' % (r.ff, self.UU(self.t), self.UU(self.x))
        return ret + '\n'.join(['%s %f %f' % (line[0], line[1][0], line[1][1])
                    for line in zip(self.xnametypes,zip(self.t,r.xf))])
        