import numpy as np
import numpy.linalg as la
def singular_values(ttslice,center,svd=la.svd):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    if center<0:
        center=len(ttslice)+center
    return left_singular_values(ttslice[:center+1],svd,False)+right_singular_values(ttslice[center:],svd,False)
def left_singular_values(ttslice,svd=la.svd,inplace=True):
    '''
        Assume that ttslice is in left-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[-1].shape
    car=ttslice[-1].reshape((-1,cshape[-1]))
    # svs=[svd(car,compute_uv=False)]
    svs=[]
    car=car.reshape((cshape[0],-1))
    for i in range(len(ttslice)-2,-1,-1):
        u,s,vh=svd(car,full_matrices=False)
        car=ttslice[i].reshape((-1,ttslice[i].shape[-1]))@(s[None,:]*u)
        car=car.reshape((ttslice[i].shape[0],-1))
        if inplace:
            ttslice[i+1]=vh.reshape((vh.shape[0],)+cshape[1:])
            cshape=ttslice[i].shape[:-1]+(vh.shape[0],)
        svs.append(s)
    if inplace:
        ttslice[0]=car.reshape(cshape)
    # svs.append(svd(car.reshape((ttslice[0].shape[0],-1)),compute_uv=False))
    return svs[::-1]


def right_singular_values(ttslice,svd=la.svd,inplace=True):
    '''
        Assume that ttslice is in right-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[0].shape
    car=ttslice[0].reshape((cshape[0],-1))
    # svs=[svd(car,compute_uv=False)]
    svs=[]
    car=car.reshape((-1,cshape[-1]))
    for i in range(1,len(ttslice)):
        u,s,vh=svd(car,full_matrices=False)
        car=(s[:,None]*vh)@ttslice[i].reshape((ttslice[i].shape[0],-1))
        car=car.reshape((-1,ttslice[i].shape[-1]))
        if inplace:
            ttslice[i-1]=u.reshape(cshape[:-1]+(u.shape[-1],))
            cshape=(u.shape[-1],)+ttslice[i].shape[1:]
        svs.append(s)
    if inplace:
        ttslice[-1]=car.reshape(cshape)
    # svs.append(svd(car.reshape((-1,ttslice[-1].shape[-1])),compute_uv=False))
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
            ttslice[o]=(1/svs[o-1])[tuple(sh)]*ttslice[o]
            sh=[None for i in range(ttslice[o].ndim)]
            sh[-1]=slice(None)
            ttslice[o-1]=svs[o-1][tuple(sh)]*ttslice[o-1]
    elif newcenter>oldcenter:
        for o in range(oldcenter,newcenter):
            sh=[None for i in range(ttslice[o].ndim)]
            sh[-1]=slice(None)
            ttslice[o]=(1/svs[o])[tuple(sh)]*ttslice[o]
            # mm=ttslice[o].reshape((-1,ttslice[o].shape[-1]))
            # if not np.allclose(mm.T.conj()@mm,np.eye(mm.shape[1],like=mm)):
            #     assert False
            sh=[None for i in range(ttslice[o].ndim)]
            sh[0]=slice(None)
            ttslice[o+1]=svs[o][tuple(sh)]*ttslice[o+1]
    else:
        pass #already correct
def svd_truncate(ar,chi_max=None,cutoff=0.0,compute_uv=True,full_matrices=None,svd=la.svd):
    if chi_max is None:
        chi_max=min(ar.shape)
    # if compute_uv:
    u,s,vh=svd(ar,full_matrices=False)
    norm=np.sqrt(np.sum(s.conj()*s))
    if (s/norm>cutoff).all():
        ind=chi_max
    else:
        ind=min(chi_max,np.argmin(s/norm>cutoff))
    return u[:,:ind],s[:ind],vh[:ind,:]
    # else:
    #     s=svd(ar,compute_uv=False)
    #     if (s>cutoff).all():
    #         ind=chi_max
    #     else:
    #         ind=min(chi_max,np.argmin(s>cutoff))
    #     return s[:ind]
    # return svd(ar,full_matrices=False,compute_uv=compute_uv)

def left_truncate_svd(ttslice,chi_max=None,cutoff=0.0,svd=la.svd):
    def _svd(x,compute_uv=True,full_matrices=False):
        return svd_truncate(x,chi_max,cutoff,compute_uv,full_matrices,svd)
    return left_singular_values(ttslice,_svd)

def right_truncate_svd(ttslice,chi_max=None,cutoff=0.0,svd=la.svd):
    def _svd(x,compute_uv=True,full_matrices=False):
        return svd_truncate(x,chi_max,cutoff,compute_uv,full_matrices,svd)
    return right_singular_values(ttslice,_svd)
