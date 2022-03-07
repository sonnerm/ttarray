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

@implement_function
def empty_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return empty(shape,dtype,cluster,chi,prototype)


@implement_function("zeros","array")
def zeros(shape,dtype=np.float64,cluster=None,chi=1,like=None):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainArray.frommatrices(ms)

@implement_function("zeros","slice")
def zeros_slice(shape,dtype=np.float64,cluster=None,chi=1):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    return TensorTrainSlice.frommatrices(ms)

@implement_function
def zeros_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    if dtype is None:
        dtype=prototype.dtype
    if isinstance(prototype,TensorTrainArray):
        if shape is None:
            shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
        else:
            cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
        return zeros(shape,dtype,cluster,chi)
    elif isinstance(prototype,TensorTrainSlice):
        if shape is None:
            shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
        else:
            cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
        return zeros_slice(shape,dtype,cluster,chi)
    else:
        return NotImplemented

@implement_function("ones","array")
def ones(shape,dtype=np.float64,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    ms=[np.ones([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for i in range(0,len(ms)-1):
        ms[i]=ms[i]/chi[i+1]
    return TensorTrainArray.frommatrices(ms)

@implement_function("ones","slice")
def ones_slice(shape,dtype=np.float64,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    ms=[np.ones([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for i in range(0,len(ms)-1):
        ms[i]=ms[i]/chi[i+1]
    return TensorTrainSlice.frommatrices(ms)

@implement_function
def ones_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return ones(shape,dtype,cluster,chi,prototype)

@implement_function("full","array")
def full(shape,fill_value,dtype=None,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    if dtype is None:
        dtype=np.array(fill_value).dtype
    ms=[np.ones([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for i in range(0,len(ms)-1):
        ms[i]=ms[i]/chi[i+1]
    ms[-1]*=fill_value
    return TensorTrainArray.frommatrices(ms)
@implement_function("full","slice")
def full_slice(shape,fill_value,dtype=None,cluster=None,chi=1,*,order=None):
    cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    if dtype is None:
        dtype=np.array(fill_value).dtype
    ms=[np.ones([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    for i in range(0,len(ms)-1):
        ms[i]=ms[i]/chi[i+1]
    ms[-1]*=fill_value
    return TensorTrainSlice.frommatrices(ms)

@implement_function
def full_like(prototype, fill_value, dtype=None, shape=None, cluster=None, chi=None):
    pass

@implement_function
def eye(N, M=None, k=0, dtype=np.float64, cluster=None,chi=1, like=None):
    if k!=0:
        raise NotImplementedError("Diagonal mps with k!=0 are not yet supported")
    raise NotImplementedError("Not yet")
@implement_function
def identity(n,dtype=None,cluster=None,chi=1, like=None):
    return eye(N=n,dtype=dtype,cluster=cluster,chi=chi,like=like)

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
