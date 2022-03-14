import numpy.linalg as la
import numpy as np
def is_canonical(ttslice,center,eps=1e-8):
    '''
        Checks whether ttslice is in canonical form with a specific orthogonality center
        and tolerance eps
    '''
    if center<0:
        center=len(ttslice)+center
    return is_left_canonical(ttslice[:center],eps) and is_right_canonical(ttslice[center+1:],eps)

def is_right_canonical(ttslice,eps=1e-8):
    for m in ttslice[1:]:
        mm=m.reshape((m.shape[0],-1))
        if not np.allclose(mm@mm.T.conj(),np.eye(mm.shape[0],like=mm),atol=eps):
            return False
    return True
def is_left_canonical(ttslice,eps=1e-8):
    for m in ttslice[:-1]:
        mm=m.reshape((-1,m.shape[-1]))
        if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm),atol=eps):
            return False
    return True

def canonicalize(ttslice,center,qr=la.qr):
    if center<0:
        center=len(ttslice)+center
    left_canonicalize(ttslice[:center+1],qr)
    right_canonicalize(ttslice[center:],qr)
def shift_orthogonality_center(ttslice,oldcenter,newcenter,qr=la.qr):
    '''
        Shift the orthogonality center by performing qr decompositions step by step
    '''
    if oldcenter<0:
        oldcenter=len(ttslice)+oldcenter
    if newcenter<0:
        newcenter=len(ttslice)+newcenter
    if oldcenter>newcenter:
        left_canonicalize(ttslice[oldcenter:newcenter+1],qr=qr)
    elif newcenter>oldcenter:
        right_canonicalize(ttslice[oldcenter:newcenter+1],qr=qr)
def left_canonicalize(ttslice,qr=la.qr):
    '''
        Bring ttslice to left canonical form with center, inplace
    '''
    car=ttslice[0].reshape((-1,ttslice[0].shape[-1]))
    cshape=ttslice[0].shape
    for i in range(1,len(ttslice)):
        nshape=ttslice[i].shape
        car=np.tensordot(car,np.reshape(ttslice[i],(nshape[0],-1)),(-1,0))
        q,r=qr(car)
        ttslice[i-1]=np.reshape(q,cshape)
        car=np.reshape(r,(-1,nshape[-1]))
        cshape=nshape
    ttslice[-1]=np.reshape(car,cshape)

def right_canonicalize(ttslice,qr=la.qr):
    def rq(x):
        q,r=qr(x.T)
        return r.T,q.T
    car=ttslice[-1].reshape((ttslice[-1].shape[0],-1))
    cshape=ttslice[-1].shape
    for i in range(len(ttslice)-2,-1,-1):
        nshape=ttslice[i].shape
        print(np.reshape(ttslice[i],(-1,nshape[-1])).shape,car.shape)
        car=np.tensordot(np.reshape(ttslice[i],(-1,nshape[-1])),car,(1,0))
        r,q=rq(car)
        ttslice[i+1]=np.reshape(q,cshape)
        car=np.reshape(r,(nshape[0],-1))
        cshape=nshape
    ttslice[0]=np.reshape(car,cshape)


def singular_values(ttslice,center,svd=la.svd):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    return left_singular_values(ttslice[:center+1])+right_singular_values(ttslice[center:])
def left_singular_values(ttslice,svd=la.svd):
    '''
        Assume that ttslice is in right-canonical form, extract singular values
    '''
    pass

def right_singular_values(ttslice,svd=la.svd):
    '''
        Assume that ttslice is in left-canonical form, extract singular values
    '''
    pass
def shift_orthogonality_center_with_singular_values(ttslice,svs,oldcenter,newcenter):
    '''
        Uses precomputed singular values to shift the orthogonality center fast
    '''
    pass
