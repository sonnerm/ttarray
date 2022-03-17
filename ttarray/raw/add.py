import numpy as np
def add_ttslice(ttslice1,ttslice2):
    '''
        Pointwise add ttslices with same boundaries and clusters
        \\chi' = \\chi_1+\\chi_2
    '''
    if len(ttslice1)==1:
        return [ttslice1[0]+ttslice2[0]]
    ttslicen=[np.concatenate([ttslice1[0],ttslice2[0]],axis=-1)]
    for t1,t2 in zip(ttslice1[1:-1],ttslice2[1:-1]):
        tt1=np.concatenate([t1,np.zeros_like(t2)],axis=0)
        tt2=np.concatenate([np.zeros_like(t1),t2],axis=0)
        ttslicen.append(np.concatenate([tt1,tt2],axis=-1))
    ttslicen.append(np.concatenate([ttslice1[-1],ttslice2[-1]],axis=0))
    return ttslicen
