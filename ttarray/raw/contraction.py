import numpy as np
def tensordot(ttslice1,ttslice2,axes):
    '''
        Contract two tensor train with the same clustering along the given axis
    '''
    maxes=tuple(i+1 for i in axes[0]),tuple(i+1 for i in axes[1])
    r1=len(ttslice1[0].shape)-2-len(axes[0])
    r2=len(ttslice2[0].shape)-2-len(axes[1])
    tp=[0,r1+2]+list(range(1,r1+1))+list(range(r1+3,r1+r2+3))+[r1+1,r1+r2+3]
    ret=[np.tensordot(t1,t2,maxes).transpose(tp) for t1,t2 in zip(ttslice1,ttslice2)]
    ret=[np.reshape(r,(r.shape[0]*r.shape[1],)+r.shape[2:-2]+(r.shape[-1]*r.shape[-2],)) for r in ret]
    return ret

def einsum(stri,*args):
    pass
