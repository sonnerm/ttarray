import numpy as np
import scipy.linalg as la
def svd_stable(a):
    try:
        u,s,vh=la.svd(a,full_matrices=False,compute_uv=True,lapack_driver="gesdd")
        if not np.allclose((u*s)@vh,a):
            import warnings
            warnings.warn("gesdd gives incorrect results; falling back to gesvd")
            return la.svd(a,full_matrices=False,compute_uv=True,lapack_driver="gesvd")
        return u,s,vh
    except la.LinAlgError:
        import warnings
        warnings.warn("gesdd did not converge; falling back to gesvd")
        return la.svd(a,full_matrices=False,compute_uv=True,lapack_driver="gesvd")
def singular_values(ttslice,center,svd=svd_stable):
    '''
        Assume ttslice is canonical and extract the singular values
    '''
    if center<0:
        center=len(ttslice)+center
    return left_singular_values(ttslice[:center+1],svd,False)+right_singular_values(ttslice[center:],svd,False)
def left_singular_values(ttslice,svd=svd_stable,inplace=True):
    '''
        Assume that ttslice is in left-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[-1].shape
    car=ttslice[-1].reshape((-1,cshape[-1]))
    svs=[]
    car=car.reshape((cshape[0],-1))
    for i in range(len(ttslice)-2,-1,-1):
        u,s,vh=svd(car)
        car=ttslice[i].reshape((-1,ttslice[i].shape[-1]))@(s[None,:]*u)
        car=car.reshape((ttslice[i].shape[0],-1))
        if inplace:
            ttslice[i+1]=vh.reshape((vh.shape[0],)+cshape[1:])
            cshape=ttslice[i].shape[:-1]+(vh.shape[0],)
        svs.append(s)
    if inplace:
        ttslice[0]=car.reshape(cshape)
    return svs[::-1]


def right_singular_values(ttslice,svd=svd_stable,inplace=True):
    '''
        Assume that ttslice is in right-canonical form, extract singular values
    '''
    if len(ttslice)==0:
        return []
    cshape=ttslice[0].shape
    car=ttslice[0].reshape((cshape[0],-1))
    svs=[]
    car=car.reshape((-1,cshape[-1]))
    for i in range(1,len(ttslice)):
        u,s,vh=svd(car)
        car=(s[:,None]*vh)@ttslice[i].reshape((ttslice[i].shape[0],-1))
        car=car.reshape((-1,ttslice[i].shape[-1]))
        if inplace:
            ttslice[i-1]=u.reshape(cshape[:-1]+(u.shape[-1],))
            cshape=(u.shape[-1],)+ttslice[i].shape[1:]
        svs.append(s)
    if inplace:
        ttslice[-1]=car.reshape(cshape)
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
def svd_truncate(ar,chi_max=None,cutoff=0.0,svd=svd_stable):
    if chi_max is None:
        chi_max=min(ar.shape)
    u,s,vh=svd(ar)
    norm=np.sqrt(np.sum(s.conj()*s))
    if (s/norm>cutoff).all():
        ind=chi_max
    else:
        ind=min(chi_max,np.argmin(s/norm>cutoff))
    return u[:,:ind],s[:ind],vh[:ind,:]

def left_truncate_svd(ttslice,chi_max=None,cutoff=0.0,svd=svd_stable):
    def _svd(x):
        return svd_truncate(x,chi_max,cutoff,svd)
    return left_singular_values(ttslice,_svd)

def right_truncate_svd(ttslice,chi_max=None,cutoff=0.0,svd=svd_stable):
    def _svd(x):
        return svd_truncate(x,chi_max,cutoff,svd)
    return right_singular_values(ttslice,_svd)
