import numpy.linalg as la
import numpy as np
def is_canonical(ttslice,center=-1,eps=1e-8):
    '''
        Checks whether ttslice is in canonical form with a specific orthogonality center
        and tolerance eps
    '''
    for m in ttslice[:center]:
        mm=m.reshape((-1,m.shape[-1]))
        if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm),atol=eps):
            return False
    if center!=-1:
        for m in ttslice[center+1:]:
            mm=m.reshape((m.shape[0],-1))
            if not np.allclose(mm@mm.T.conj(),np.eye(mm.shape[0],like=mm),atol=eps):
                return False
    return True

def canonicalize(ttslice,qr=la.qr,center=-1):
    '''
        Bring ttslice to canonical form with center, inplace
    '''
    def rq(x):
        q,r=qr(x.T)
        return r.T,q.T
    if center<0:
        center=len(ttslice)+center
    cshape=car.shape
    car=ttslice[0].reshape((-1,car.shape[-1]))
    for i in range(1,center):
        nshape=ttslice[i].shape
        car=np.tensordot(car,np.reshape(ttslice[i],(nshape[0],-1)),(-1,0))
        q,r=qr(car)
        ttslice[i-1]=np.reshape(q,cshape)
        car=r
        cshape=nshape
    ttslice[center]=np.reshape(car,cshape)
    for i in range(center+1,len(ttslice)-1):
        nshape=ttslice[i].shape
        car=np.tensordot(car,np.reshape(ttslice[i],(nshape[0],-1)),(-1,0))
        q,r=rq(car)
        ttslice[i-1]=np.reshape(r,cshape)
        car=q
        cshape=nshape



def singular_values(ttslice,center,svd=la.svd):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    pass
def shift_orthogonality_center_with_singular_values(ttslice,svs,oldcenter,newcenter):
    '''
        Uses precomputed singular values to shift the orthogonality center fast
    '''
    pass
def shift_orthogonality_center(ttslice,oldcenter,newcenter,qr=la.qr):
    '''
        Shift the orthogonality center by performing qr decompositions step by step
    '''
    pass
