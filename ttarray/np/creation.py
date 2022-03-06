from .. import TensorTrainArray,TensorTrainSlice
from ..dispatch import implement_function
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


@implement_function(np.empty)
def empty(shape,dtype=np.float64, cluster=None,chi=1, like=None):
    '''
        Create an empty TensorTrainSlice or TensorTrainArray
    '''
    if isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    elif isinstance(like,TensorTrainArray) or like is None:
        cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    else:
        return np.empty(shape,dtype,like=like)
    ms=[np.empty([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    if like is None:
        return TensorTrainArray.frommatrices(ms)
    if like is TensorTrainSlice:
        return TensorTrainSlice.frommatrices(ms)
    else:
        return like.__class__.frommatrices(ms)
def empty_slice(shape,dtype=np.float64, cluster=None,chi=1):
    return empty(shape,dtype,cluster,chi,like=TensorTrainSlice)

@implement_function(np.empty_like)
def empty_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return empty(shape,dtype,cluster,chi,prototype)


@implement_function(np.zeros)
def zeros(shape,dtype=np.float64,cluster=None,chi=1,like=None):
    if isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    elif isinstance(like,TensorTrainArray) or like is None:
        cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    else:
        return np.zeros(shape,dtype,like=like) # like is guaranteed not None
    ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
        # not ideal, maybe numpy changes that like arg cannot be None
    if like is None:
        return TensorTrainArray.frommatrices(ms)
    if like is TensorTrainSlice:
        return TensorTrainSlice.frommatrices(ms)
    else:
        return like.__class__.frommatrices(ms)

def zeros_slice(shape,dtype=np.float64,cluster=None,chi=1):
    return zeros(shape,dtype,cluster,chi,like=TensorTrainSlice)

@implement_function(np.zeros_like)
def zeros_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return zeros(shape,dtype,cluster,chi,prototype)

@implement_function(np.ones)
def ones(shape,dtype=np.float64,cluster=None,chi=1,like=None):
    if isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        cluster,chi=_get_cluster_chi_slice(shape,cluster,chi)
    elif isinstance(like,TensorTrainArray) or like is None:
        cluster,chi=_get_cluster_chi_array(shape,cluster,chi)
    else:
        return np.ones(shape,dtype,like=like) # like is guaranteed not None
    ms=[np.ones([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
        # not ideal, maybe numpy changes that like arg cannot be None
    if like is None:
        return TensorTrainArray.frommatrices(ms)
    if like is TensorTrainSlice:
        return TensorTrainSlice.frommatrices(ms)
    else:
        return like.__class__.frommatrices(ms)
@implement_function(np.ones_like)
def ones_like(prototype, dtype=None, shape=None, cluster=None, chi=None):
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return ones(shape,dtype,cluster,chi,prototype)

@implement_function(np.full)
def full(shape,fill_value,dtype=None,cluster=None,chi=1,like=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if isinstance(like,TensorTrainSlice):
        if len(shape<2):
            raise ValueError("TensorTrainSlice has at least 2 dimensions.")
        else:
            chi=[shape[0]]+chi+[shape[-1]]
            shape=shape[1:-1]
    elif isinstance(like,TensorTrainArray):
        chi=[1]+chi+[1]
    else:
        return NotImplemented
    val=fill_value**(1/(len(chi)+1))
    ms=[np.full(s,val,dtype,order)/s[0]/s[-1] for s in zip(chi[1:],*cluster,chi[:-1])]
    return like.__class__.from_matrix_list(ms)

@implement_function(np.full_like)
def full_like(prototype, fill_value, dtype=None, shape=None, cluster=None, chi=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    return full(shape,fill_value,dtype,cluster,chi,prototype)

@implement_function(np.eye)
def eye(N, M=None, k=0, dtype=np.float64, cluster=None,chi=1, like=None):
    if k!=0:
        raise NotImplementedError("Diagonal mps with k!=0 are not yet supported")
    raise NotImplementedError("Not yet")
@implement_function(np.identity)
def identity(n,dtype=None,cluster=None,chi=1, like=None):
    return eye(N=n,dtype=dtype,cluster=cluster,chi=chi,like=like)

@implement_function(np.array)
def array(ar, dtype=None, cluster=None, copy=True, ndmin=0, like=None):
    '''
        like defaults to TensorTrainArray
    '''
    if isinstance(like,TensorTrainArray) or like is None:
        if isinstance(ar,TensorTrainArray):
            #copy if necessary
            pass
        elif isinstance(ar,TensorTrainSlice):
            #just recast, clustering is then a bit weird
            pass
        else:
            ar=ar[None,...,None]
            return like.__class__(array_to_ttslice(ar,find_balanced_cluster(ar.shape),la.qr))
    elif isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        if isinstance(ar,TensorTrainSlice):
            #copy if necessary
            pass
        elif isinstance(ar,TensorTrainArray):
            #recluster then recast
            pass
        return like.__class__(array_to_ttslice(ar,find_balanced_cluster(ar.shape),la.qr))
    else:
        return np.array(ar,dtype=dtype,copy=copy,ndim=ndim,like=like) #fallback to numpy
def slice(ar, dtype=None, cluster=None, copy=True, ndmin=0):
    return array(ar,dtype,cluster,copy,ndim,like=TensorTrainSlice)
@implement_function(np.asarray)
def asarray(ar, dtype=None,cluster=None,like=None):
    array(ar,dtype,cluster=cluster,like=like,copy=False)

@implement_function(np.asanyarray)
def asanyarray(ar, dtype=None,cluster=None,like=None):
    array(ar,dtype,cluster=cluster,like=like,copy=False)

@implement_function(np.frombuffer)
def frombuffer(buffer, dtype=float, count=- 1, offset=0, cluster=None, like=None):
    array(np.frombuffer(buffer,dtype,count,offset),dtype=dtype,cluster=cluster,like=like,copy=False)
