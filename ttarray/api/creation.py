from .. import TensorTrainArray,TensorTrainSlice
from ..ttarray import implement_function
from ..raw import find_balanced_cluster
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

@implement_function()
def eye(N, M=None, k=0, dtype=np.float64, cluster=None,chi=1):
    if k!=0:
        raise NotImplementedError("Diagonal mps with k!=0 are not yet supported")
    raise NotImplementedError("Not yet")
@implement_function()
def identity(n,dtype=None,cluster=None,chi=1):
    return eye(N=n,dtype=dtype,cluster=cluster,chi=chi)

@implement_function("array","array")
def array(ar, dtype=None, cluster=None, copy=True, ndmin=0):
    if isinstance(ar,TensorTrainArray):
        #copy if necessary
        pass
    elif isinstance(ar,TensorTrainSlice):
        #just recast, clustering is then a bit weird
        pass
    else:
        ar=ar[None,...,None]
        return like.__class__(array_to_ttslice(ar,find_balanced_cluster(ar.shape),la.qr))
@implement_function("array","slice")
def slice(ar, dtype=None, cluster=None, copy=True, ndmin=0):
    if isinstance(ar,TensorTrainSlice):
        #copy if necessary
        pass
    elif isinstance(ar,TensorTrainArray):
        #recluster then recast
        pass
    else:
        pass

@implement_function("asarray","array")
def asarray(ar, dtype=None,cluster=None):
    array(ar,dtype,cluster=cluster,copy=False)
@implement_function("asarray","slice")
def asslice(ar, dtype=None,cluster=None,like=None):
    slice(ar,dtype,cluster=cluster,copy=False)

@implement_function(np.asanyarray)
def asanyarray(ar, dtype=None,cluster=None,like=None):
    array(ar,dtype,cluster=cluster,like=like,copy=False)

@implement_function(np.frombuffer)
def frombuffer(buffer, dtype=float, count=- 1, offset=0, cluster=None, like=None):
    array(np.frombuffer(buffer,dtype,count,offset),dtype=dtype,cluster=cluster,like=like,copy=False)
