import numpy as np
def contract(ttslice1,ttslice2,axis):
    '''
        Contract two tensor train with the same clustering along the given axis
    '''
    def _contract_tensor(t1,t2,axis):
        pass
    return [_contract_tensor(t1,t2,axis) for t1,t2 in zip(ttslice1,ttslice2)]
