from cge_tools import Series, rsum, cobbdouglas
import pandas as pd

class Calibration:
    def __init__(self, parameter):
        self.parameter = parameter
        self.X0 = {}
        for consumer in parameter.consumers:
            print parameter.sam.sam
            self.X0[consumer] = parameter.sam.inputs(
            											inputs=parameter.industries, 
            											to=consumer
            					)
            # G0 = sam.field_by_rows("gov", ii + ih)
        self.F0 = self.parameter.sam.sub_matrix(
        											rows=parameter.factors, 
        											columns=parameter.industries
        		  )
        self.Z0 = self.parameter.sam.sum_by_rows(parameter.industries)
        self.FF = self.parameter.sam.sum_by_rows(parameter.factors)
        self.alpha = Series.like(self.X0['HH'])
        self.sX0 = rsum(self.X0['HH'])
        for i in xrange(len(parameter.industries)):
            self.alpha[i] = self.X0['HH'][i] / self.sX0
        sF0 = self.F0.sum()        
        self.beta = pd.DataFrame(
        							index=parameter.factors, 
        							columns=parameter.industries
        			)
        for h in parameter.factors:
            for j in parameter.industries:
                self.beta[j][h] = self.F0[j][h] / sF0[j]
        self.b = Series.like(self.Z0)
        for j, J in enumerate(parameter.industries):
            self.b[j] = self.Z0[j] / cobbdouglas(
            										self.F0, 
            										self.beta,  
            										industry=J, 
            										factors=parameter.factors
            						)

    def __str__(self):
        return 'X0%sF0\n%s\nZ0%s\nFF\n%s\nalpha\n%s\nbeta\n%s\nb\n%s' % (
        self.X0, self.F0, self.Z0, self.FF, self.alpha, self.beta, self.b)

    def values(self):
        return self.X0, self.F0, self.Z0, self.FF, self.alpha, self.beta, self.b        