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
        sslice=ttslice[newcenter:oldcenter+1]
        right_canonicalize(sslice,qr=qr)
        ttslice[newcenter:oldcenter+1]=sslice
    elif newcenter>oldcenter:
        sslice=ttslice[oldcenter:newcenter+1]
        left_canonicalize(sslice,qr=qr)
        ttslice[oldcenter:newcenter+1]=sslice
def left_canonicalize(ttslice,qr=la.qr):
    '''
        Bring ttslice to left canonical form with center, inplace
    '''
    car=ttslice[0].reshape((-1,ttslice[0].shape[-1]))
    cshape=ttslice[0].shape
    for i in range(1,len(ttslice)):
        nshape=ttslice[i].shape
        car=car@np.reshape(ttslice[i],(nshape[0],-1))
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
        car=np.reshape(ttslice[i],(-1,nshape[-1]))@car
        r,q=rq(car)
        ttslice[i+1]=np.reshape(q,cshape)
        car=np.reshape(r,(nshape[0],-1))
        cshape=nshape
    ttslice[0]=np.reshape(car,cshape)


def singular_values(ttslice,center,svd=la.svd):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    if center<0:
        center=len(ttslice)+center
    return left_singular_values(ttslice[:center])+right_singular_values(ttslice[center:])
def left_singular_values(ttslice,svd=la.svd):
    '''
        Assume that ttslice is in left-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[-1].shape
    car=ttslice[-1].reshape((-1,cshape[-1]))
    svs=[svd(car,compute_uv=False)]
    car=car.reshape((cshape[0],-1))
    for i in range(len(ttslice)-2,-1,-1):
        car=ttslice[i].reshape((-1,ttslice[i].shape[-1]))@car
        u,s,vh=svd(car)
        svs.append(s)
        car=u.reshape((ttslice[i].shape[0],-1))
    svs.append(svd(car.reshape((ttslice[0].shape[0],-1)),compute_uv=False))
    return svs[::-1]


def right_singular_values(ttslice,svd=la.svd):
    '''
        Assume that ttslice is in right-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[0].shape
    car=ttslice[0].reshape((cshape[0],-1))
    svs=[svd(car,compute_uv=False)]
    car=car.reshape((-1,cshape[-1]))
    for i in range(1,len(ttslice)):
        car=car@ttslice[i].reshape((ttslice[i].shape[0],-1))
        u,s,vh=svd(car,full_matrices=False)
        svs.append(s)
        car=(s[:,None]*vh).reshape((-1,ttslice[i].shape[-1]))
    svs.append(svd(car.reshape((-1,ttslice[-1].shape[-1])),compute_uv=False))
    return svs

def shift_orthogonality_center_with_singular_values(ttslice,oldcenter,newcenter,svs):
    '''
        Uses precomputed singular values to shift the orthogonality center fast
    '''
    if newcenter<0:
        newcenter=len(ttslice)+newcenter
    if oldcenter<0:
        oldcenter=len(ttslice)+oldcenter
    if oldcenter>newcenter:
        for o in range(oldcenter,newcenter,-1):
            sh=[None for i in range(ttslice[o].ndim)]
            sh[0]=slice(None)
            ttslice[o]=(1/svs[o])[tuple(sh)]*ttslice[o]
            sh=[None for i in range(ttslice[o].ndim)]
            sh[-1]=slice(None)
            ttslice[o-1]=svs[o][tuple(sh)]*ttslice[o-1]
    elif newcenter>oldcenter:
        for o in range(oldcenter,newcenter):
            sh=[None for i in range(ttslice[o].ndim)]
            sh[-1]=slice(None)
            ttslice[o]=(1/svs[o+1])[tuple(sh)]*ttslice[o]
            sh=[None for i in range(ttslice[o].ndim)]
            sh[0]=slice(None)
            ttslice[o+1]=svs[o+1][tuple(sh)]*ttslice[o+1]
    else:
        pass #already correct
