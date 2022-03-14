import numpy as np
def multiply_ttslice(ttslice1,ttslice2):
    '''
        Pointwise multiply two ttslices with the same boundaries and clusters
    '''
    if len(ttslice1)==1:
        return [ttslice1[0]*ttslice2[0]] # easy
    cshape=ttslice1[0].shape
    t1=ttslice1[0].reshape((-1,ttslice1[0].shape[-1]))
    t2=ttslice2[0].reshape((-1,ttslice2[0].shape[-1]))
    ttslicen=[np.einsum("ab,ac->abc",t1,t2).reshape(cshape[:-1]+(t1.shape[-1]*t2.shape[-1],))]
    for t1,t2 in zip(ttslice1[1:-1],ttslice2[1:-1]):
        cshape=t1.shape
        t1=t1.reshape((t1.shape[0],-1,t1.shape[-1]))
        t2=t2.reshape((t2.shape[0],-1,t2.shape[-1]))
        ttslicen.append(np.einsum("abc,dbe->adbce",t1,t2).reshape((t1.shape[0]*t2.shape[0],)+cshape[1:-1]+(t1.shape[-1]*t2.shape[-1],)))
    cshape=ttslice1[-1].shape
    t1=ttslice1[-1].reshape((ttslice1[-1].shape[0],-1))
    t2=ttslice2[-1].reshape((ttslice2[-1].shape[0],-1))
    ttslicen.append(np.einsum("ab,cb->acb",t1,t2).reshape((t1.shape[0]*t2.shape[0],)+cshape[1:]))
    return ttslicen
