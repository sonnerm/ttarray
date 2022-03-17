import numpy as np
import numpy.linalg as la
def singular_values(ttslice,center,svd=la.svd):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    if center<0:
        center=len(ttslice)+center
    return left_singular_values(ttslice[:center+1])[:-2]+right_singular_values(ttslice[center:])
def left_singular_values(ttslice,svd=la.svd,inplace=False):
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
        u,s,vh=svd(car,full_matrices=False)
        if inplace:
            ttslice[i+1]=vh.reshape((vh.shape[0],)+cshape[1:])
            cshape=ttslice[i].shape[:-1]+(vh.shape[0],)
        svs.append(s)
        car=(s[None,:]*u).reshape((ttslice[i].shape[0],-1))
    if inplace:
        ttslice[0]=car.reshape(cshape)
    svs.append(svd(car.reshape((ttslice[0].shape[0],-1)),compute_uv=False))
    return svs[::-1]


def right_singular_values(ttslice,svd=la.svd,inplace=False):
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
        if inplace:
            ttslice[i-1]=u.reshape(cshape[:-1]+(u.shape[-1],))
            cshape=(u.shape[-1],)+ttslice[i].shape[1:]
        svs.append(s)
        car=(s[:,None]*vh).reshape((-1,ttslice[i].shape[-1]))
    if inplace:
        ttslice[-1]=car.reshape(cshape)
    svs.append(svd(car.reshape((-1,ttslice[-1].shape[-1])),compute_uv=False))
    return svs

def shift_orthogonality_center_with_singular_values(ttslice,oldcenter,newcenter,svs):
    '''
        Uses precomputed singular values to shift the orthogonality center fast
        * only works if ttslice is in svd-canonical form!!! *
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
            # mm=ttslice[o].reshape((-1,ttslice[o].shape[-1]))
            # if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm)):
            #     assert False
            sh=[None for i in range(ttslice[o].ndim)]
            sh[0]=slice(None)
            ttslice[o+1]=svs[o+1][tuple(sh)]*ttslice[o+1]
    else:
        pass #already correct
def svd_truncate(ar,chi_max=None,cutoff=0.0,compute_uv=True,full_matrices=None,svd=la.svd):
    if chi_max is None:
        chi_max=min(ar.shape)
    if compute_uv:
        u,s,vh=svd(ar,full_matrices=False)
        ind=min(chi_max,np.argmax(s<cutoff)-1)
        return u[:,:ind],s[:ind],vh[:ind,:]
    else:
        s=svd(ar,compute_uv=False)
        ind=min(chi_max,np.argmax(s<cutoff)-1)
        return s[:ind]

def left_truncate_svd(ttslice,chi_max=None,cutoff=None,svd=la.svd):
    return left_singular_values(ttslice,svd=lambda x:svd_truncate(x,chi_max,cutoff,svd))

def right_truncate_svd(ttslice,svd=la.svd):
    return right_singular_values(ttslice,svd=lambda x:svd_truncate(x,chi_max,cutoff,svd))
