from sam import Sam


class Parameter:
    def __init__(self, index, industries, factors, consumers, table=None):
        self.index = index
        self.industries = industries
        self.factors = factors
        self.consumers = consumers
        self.sam = Sam(table=table, index=index, columns=index)


    def __str__(self):
        return 'sam\n%s\nindustries\n%s\nfactors\n%s\nconsumers\n%s' % (
            self.sam, self.industries, self.factors, self.consumers)