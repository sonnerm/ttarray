import numpy.linalg as la
import numpy as np
def is_canonical(ttslice,center,eps=1e-14):
    '''
        Checks whether ttslice is in canonical form with a specific orthogonality center
        and tolerance eps
    '''
    if center<0:
        center=len(ttslice)+center
    return is_left_canonical(ttslice[:center+1],eps) and is_right_canonical(ttslice[center:],eps)

def is_right_canonical(ttslice,eps=1e-14):
    for m in ttslice[1:]:
        mm=m.reshape((m.shape[0],-1))
        if not np.allclose(mm@mm.T.conj(),np.eye(mm.shape[0],like=mm),atol=eps):
            return False
    return True
def is_left_canonical(ttslice,eps=1e-14):
    for m in ttslice[:-1]:
        mm=m.reshape((-1,m.shape[-1]))
        if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm),atol=eps):
            return False
    return True

def canonicalize(ttslice,center,qr=la.qr):
    if center<0:
        center=len(ttslice)+center
    lh=ttslice[:center+1]
    left_canonicalize(lh,qr)
    ttslice[:center+1]=lh
    rh=ttslice[center:]
    right_canonicalize(rh,qr)
    ttslice[center:]=rh
def shift_orthogonality_center(ttslice,oldcenter,newcenter,qr=la.qr):
    '''
        Shift the orthogonality center by performing qr decompositions step by step
    '''
    if oldcenter<0:
        oldcenter=len(ttslice)+oldcenter
    if newcenter<0:
        newcenter=len(ttslice)+newcenter
    if oldcenter>newcenter:
        sslice=ttslice[newcenter:oldcenter+1]
        right_canonicalize(sslice,qr=qr)
        ttslice[newcenter:oldcenter+1]=sslice
    elif newcenter>oldcenter:
        sslice=ttslice[oldcenter:newcenter+1]
        left_canonicalize(sslice,qr=qr)
        ttslice[oldcenter:newcenter+1]=sslice
def left_canonicalize(ttslice,qr=la.qr):
    '''
        Bring ttslice to left canonical form, inplace
    '''
    car=ttslice[0].reshape((-1,ttslice[0].shape[-1]))
    cshape=ttslice[0].shape
    for i in range(1,len(ttslice)):
        nshape=ttslice[i].shape
        q,r=qr(car)
        cshape=cshape[:-1]+(q.shape[-1],)
        ttslice[i-1]=np.reshape(q,cshape)
        car=r@np.reshape(ttslice[i],(nshape[0],-1))
        car=np.reshape(car,(-1,nshape[-1]))
        cshape=nshape
        cshape=(r.shape[0],)+cshape[1:]
    ttslice[-1]=np.reshape(car,cshape)

def right_canonicalize(ttslice,qr=la.qr):
    def rq(x):
        q,r=qr(x.T)
        return r.T,q.T
    car=ttslice[-1].reshape((ttslice[-1].shape[0],-1))
    cshape=ttslice[-1].shape
    for i in range(len(ttslice)-2,-1,-1):
        nshape=ttslice[i].shape
        r,q=rq(car)
        cshape=(q.shape[0],)+cshape[1:]
        ttslice[i+1]=np.reshape(q,cshape)
        car=np.reshape(ttslice[i],(-1,nshape[-1]))@r
        car=np.reshape(car,(nshape[0],-1))
        cshape=nshape
        cshape=cshape[:-1]+(r.shape[-1],)
    ttslice[0]=np.reshape(car,cshape)
def find_orthogonality_center(ttslice,eps=1e-8):
    '''
        If ttslice is in mixed canonical form, return the orthogonality center, otherwise return None
    '''
    for i,m in enumerate(ttslice):
        mm=m.reshape((-1,m.shape[-1]))
        if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm),atol=eps):
            break
    if i==len(ttslice)-1:
        return len(ttslice)-1
    for m in ttslice[i+1:]:
        mm=m.reshape((m.shape[0],-1))
        if not np.allclose(mm@mm.T.conj(),np.eye(mm.shape[0],like=mm),atol=eps):
            return None
    return i
