from .array import TensorTrainArray
from .slice import TensorTrainSlice
from .dispatch import implement_function
from ..raw import find_balanced_cluster,trivial_decomposition
import numpy as np
def _get_cluster_chi_array(shape,cluster,chi):
    if cluster is None:
        cluster=find_balanced_cluster(shape)
    if isinstance(chi,int):
        chi=[chi]*(len(cluster)-1)
    chi=tuple([1]+list(chi)+[1])
    return cluster,chi

def _get_cluster_chi_slice(shape,cluster,chi):
    if len(shape)<2:
        raise ValueError("TensorTrainSlice has at least 2 dimensions.")
    if cluster is None:
        cluster=find_balanced_cluster(shape[1:-1])
    if isinstance(chi,int):
        chi=[chi]*(len(cluster)-1)
    chi=tuple([shape[0]]+list(chi)+[shape[-1]])
    return cluster,chi


@implement_function("empty","array")
def empty(shape,dtype=np.float64, cluster=None,chi=1):
    '''
        Create an empty TensorTrainArray
    '''
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    ms=[np.empty([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainArray.frommatrices(ms)

@implement_function("empty","slice")
def empty_slice(shape,dtype=np.float64, cluster=None,chi=1):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    ms=[np.empty([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainSlice.frommatrices(ms)

@implement_function()
def empty_like(prototype, dtype=None, shape=None, cluster=None, chi=1,*,order=None,subok=None):
    if dtype is None:
        dtype=prototype.dtype
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if isinstance(prototype,TensorTrainArray):
        return empty(shape,dtype,cluster,chi)
    elif isinstance(prototype,TensorTrainSlice):
        return empty_slice(shape,dtype,cluster,chi)
    else:
        return NotImplemented


@implement_function("zeros","array")
def zeros(shape,dtype=np.float64,cluster=None,chi=1):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainArray.frommatrices(ms)

@implement_function("zeros","slice")
def zeros_slice(shape,dtype=np.float64,cluster=None,chi=1):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainSlice.frommatrices(ms)

@implement_function()
def zeros_like(prototype, dtype=None, shape=None, cluster=None, chi=1,*,order=None,subok=None):
    if dtype is None:
        dtype=prototype.dtype
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if isinstance(prototype,TensorTrainArray):
        return zeros(shape,dtype,cluster,chi)
    elif isinstance(prototype,TensorTrainSlice):
        return zeros_slice(shape,dtype,cluster,chi)
    else:
        return NotImplemented

@implement_function("ones","array")
def ones(shape,dtype=np.float64,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for m in ms:
        m[0,...,0]=np.ones(m.shape[1:-1],dtype)
    return TensorTrainArray.frommatrices(ms)

@implement_function("ones","slice")
def ones_slice(shape,dtype=np.float64,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    if len(ms)==1:
        ms[0]=np.ones(ms[0].shape,dtype)
    else:
        ms[0][...,0]=np.ones(ms[0].shape[:-1],dtype)
        ms[-1][0,...]=np.ones(ms[-1].shape[1:],dtype)
    for m in ms[1:-1]:
        m[0,...,0]=np.ones(m.shape[1:-1],dtype)
    return TensorTrainSlice.frommatrices(ms)

@implement_function()
def ones_like(prototype, dtype=None, shape=None, cluster=None, chi=1,*,order=None,subok=None):
    if dtype is None:
        dtype=prototype.dtype
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if isinstance(prototype,TensorTrainArray):
        return ones(shape,dtype,cluster,chi)
    elif isinstance(prototype,TensorTrainSlice):
        return ones_slice(shape,dtype,cluster,chi)
    else:
        return NotImplemented

@implement_function("full","array")
def full(shape,fill_value,dtype=None,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    if dtype is None:
        dtype=np.array(fill_value).dtype
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for m in ms:
        m[0,...,0]=np.ones(m.shape[1:-1],dtype)
    ms[-1]*=fill_value
    return TensorTrainArray.frommatrices(ms)
@implement_function("full","slice")
def full_slice(shape,fill_value,dtype=None,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    if dtype is None:
        dtype=np.array(fill_value).dtype
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    if len(ms)==1:
        ms[0]=np.ones(ms[0].shape,dtype)
    else:
        ms[0][...,0]=np.ones(ms[0].shape[:-1],dtype)
        ms[-1][0,...]=np.ones(ms[-1].shape[1:],dtype)
    for m in ms[1:-1]:
        m[0,...,0]=np.ones(m.shape[1:-1],dtype)
    ms[-1]*=fill_value
    return TensorTrainSlice.frommatrices(ms)

@implement_function()
def full_like(prototype, fill_value, dtype=None, shape=None, cluster=None, chi=1):
    if dtype is None:
        dtype=prototype.dtype
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if isinstance(prototype,TensorTrainArray):
        return full(shape,fill_value,dtype,cluster,chi)
    elif isinstance(prototype,TensorTrainSlice):
        return full_slice(shape,fill_value,dtype,cluster,chi)
    else:
        return NotImplemented

@implement_function("eye","array")
def eye(N, M=None, k=0, dtype=np.float64, cluster=None):
    if M!=None:
        raise NotImplementedError("not implemented yet ...")
    return diag(ones((N,),dtype=dtype,cluster=cluster),k)
@implement_function("identity","array")
def identity(n,dtype=None,cluster=None):
    return eye(N=n,dtype=dtype,cluster=cluster)
@implement_function("diag","array")
def diag(v,k=0):
    if k!=0:
        raise NotImplementedError("off diagonal not yet supported")
    if not isinstance(v,TensorTrainArray):
        return NotImplemented
    if len(v.shape)==1:
        return frommatrices_unchecked(np.einsum("abd->abbd",x) for x in v.tomatrices_unchecked())
    if len(v.shape)==2:
        return frommatrices_unchecked(np.einsum("abbd->abd",x) for x in v.tomatrices_unchecked())
    else:
        raise ValueError("Input must be either 1- or 2-d.")



@implement_function("array","array")
def array(ar, dtype=None, cluster=None, copy=True,*,ndim=0):
    if isinstance(ar,TensorTrainArray):
        #copy if necessary, recluster if necessary
        if cluster is not None and cluster!=ar.cluster:
            return ar.recluster(cluster,copy=True)
        elif copy:
            return ar.copy()
        else:
            return ar
    elif isinstance(ar,TensorTrainSlice):
        #just recast, clustering is then a bit weird, recluster afterwards
        arm=ar.tomatrices()
        if not copy:
            arm[0]=arm[0][None,...]
        else:
            pass
        ret=TensorTrainArray.frommatrices(arm)
        if cluster is not None:
            ret.recluster(cluster)
        return ret
    else:
        return TensorTrainArray.fromdense(ar,dtype,cluster)
@implement_function("array","slice")
def slice(ar, dtype=None, cluster=None, copy=True,*,ndim=0):
    if isinstance(ar,TensorTrainSlice):
        if cluster is not None and cluster!=ar.cluster:
            return ar.recluster(cluster,copy=True)
        elif copy:
            return ar.copy()
        else:
            return ar
    elif isinstance(ar,TensorTrainArray):
        #recluster then recast
        raise NotImplementedError("Conversion TensorTrainArray to slice not yet implemented")
    else:
        return TensorTrainSlice.fromdense(ar,dtype,cluster)
@implement_function("asarray","array")
def asarray(ar, dtype=None,cluster=None):
    return array(ar,dtype,cluster=cluster,copy=False)
@implement_function("asarray","slice")
def asslice(ar, dtype=None,cluster=None):
    return slice(ar,dtype,cluster=cluster,copy=False)

@implement_function("asanyarray","array")
def asanyarray(ar, dtype=None,cluster=None):
    return array(ar,dtype,cluster=cluster,copy=False)

@implement_function("asanyarray","slice")
def asanyslice(ar, dtype=None,cluster=None):
    return slice(ar,dtype,cluster=cluster,copy=False)

@implement_function("frombuffer","array")
def frombuffer(buffer, dtype=float, count=- 1, offset=0, cluster=None):
    return array(np.frombuffer(buffer,dtype,count,offset),dtype=dtype,cluster=cluster)

@implement_function("fromiter","array")
def fromiter(iter, dtype, count=- 1, cluster=None):
    return array(np.fromiter(iter,dtype,count),dtype=dtype,cluster=cluster)

def frommatrices(iter):
    return TensorTrainArray.frommatrices(iter)

def frommatrices_slice(iter):
    return TensorTrainSlice.frommatrices(iter)

def fromproduct(iter):
    return TensorTrainArray.frommatrices((x[None,...,None] for x in iter))

def fromproduct_slice(iter):
    return TensorTrainSlice.frommatrices((x[None,...,None] for x in iter))

@implement_function("fromfunction","array")
def fromfunction(function, shape, dtype=float, cluster=None, **kwargs):
    '''
        Should be upgraded to support ttcross eventually, so might change behavior if function is not sane
    '''
    return array(np.fromfunction(function,shape,dtype=dtype,**kwargs),dtype=dtype,cluster=cluster)

@implement_function("fromfunction","slice")
def fromfunction_slice(function, shape, dtype=float, cluster=None, **kwargs):
    '''
        Should be upgraded to support ttcross eventually, so might change behavior if function is not sane
    '''
    return slice(np.fromfunction(function,shape,dtype=dtype,**kwargs),dtype=dtype,cluster=cluster)

@implement_function()
def copy(a,*,order=None,subok=None):
    if isinstance(a,TensorTrainArray) or isinstance(a,TensorTrainSlice):
        return a.copy()
    else:
        return NotImplemented

@implement_function("arange","array")
def arange(*args, **kwargs):
    #wild hack to deal with optional arguments
    if len(args)==5:
        array(np.arange(*args[:-1],**kwargs),cluster=args[-1])
    elif "cluster" in kwargs.keys():
        cluster=kwargs["cluster"]
        del kwargs["cluster"]
        array(np.arange(*args,**kwargs),cluster=cluster)
    else:
        array(np.arange(*args,**kwargs))

@implement_function("linspace","array")
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0,cluster=None):
    array(np.linspace(todense(start),todense(stop),num,endpoint,retstep,dtype,axis),cluster=cluster)

# @implement_function("linspace","slice")
# def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0,cluster=None):
#     slice(np.linspace(todense(start),todense(stop),num,endpoint,retstep,dtype,axis),cluster=cluster)

@implement_function("logspace","array")
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0,cluster=None):
    raise NotImplementedError("not yet")

# @implement_function("logspace","slice")
# def logspace_slice(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0,cluster=None):
#     slice(np.logspace(todense(start),todense(stop),num,endpoint,base,dtype,axis),cluster=cluster)

@implement_function("geomspace","array")
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0,cluster=None):
    raise NotImplementedError("not yet")

# @implement_function("geomspace","slice")
# def geomspace_slice(start, stop, num=50, endpoint=True, dtype=None, axis=0,cluster=None):
#     slice(np.geomspace(todense(start),todense(stop),num,endpoint,dtype,axis),cluster=cluster)
# def fromdense(ar,dtype=None,cluster=None):
#     return TensorTrainArray.fromdense(ar,dtype,cluster)
# def fromdense_slice(ar,dtype,cluster):
#     return TensorTrainSlice.fromdense(ar,dtype,cluster)
def todense(ttar):
    return ttar.todense()

@implement_function("asfarray","array")
def asfarray(ttar,dtype=None):
    if not np.issubdtype(dtype,np.inexact):
        dtype=float
    return asarray(ttar,dtype=dtype)



@implement_function("asfarray","slice")
def asfslice(ttar,dtype=None):
    if not np.issubdtype(dtype,np.inexact):
        dtype=float
    return asslice(ttar,dtype=dtype)
