import numpy as np
def tensordot(ttslice1,ttslice2,axes):
    '''
        Contract two tensor train with the same clustering along the given axis
    '''
    maxes=tuple(i+1 for i in axes[0]),tuple(i+1 for i in axes[1])
    return [np.tensordot(t1,t2,maxes) for t1,t2 in zip(ttslice1,ttslice2)]
