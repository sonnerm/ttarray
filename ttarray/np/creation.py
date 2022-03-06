from .. import TensorTrainArray,TensorTrainSlice
from ..dispatch import implement_function
from ..raw import find_balanced_cluster
import numpy as np
def _get_cluster_chi(shape,cluster,chi):
    if cluster is None:
        cluster=find_balanced_cluster(shape)
    if isinstance(chi,int):
        chi=[chi]*(len(cluster)-1)
    return cluster,chi


@implement_function(np.empty)
def empty(shape,dtype=np.float64, cluster=None,chi=1, inner_like=None, like=None):
    '''
        Create an empty TensorTrainSlice or TensorTrainArray
    '''
    cluster,chi=_get_cluster_chi(shape,cluster,chi)
    if isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        if len(shape<2):
            raise ValueError("TensorTrainSlice has at least 2 dimensions.")
        else:
            chi=[shape[0]]+list(chi)+[shape[-1]]
            shape=shape[1:-1]
    elif isinstance(like,TensorTrainArray) or like is None:
        chi=[1]+list(chi)+[1]
    else:
        return np.empty(shape,dtype,like=like)
    ms=[np.empty([c1]+list(s)+[c2],dtype,like=inner_like) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    if like is None:
        return TensorTrainArray.frommatrices(ms)
    if like is TensorTrainSlice:
        return TensorTrainSlice.frommatrices(ms)
    else:
        return like.__class__.frommatrices(ms)
def empty_slice(shape,dtype=np.float64, cluster=None,chi=1, inner_like=None):
    return empty(shape,dtype,cluster,chi,inner_like,like=TensorTrainSlice)

@implement_function(np.empty_like)
def empty_like(prototype, dtype=None, shape=None, cluster=None, chi=None,inner_like=None):
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    if inner_like is None:
        inner_like=prototype.M[0]
    return empty(shape,dtype,cluster,chi,inner_like,prototype)


@implement_function(np.zeros)
def zeros(shape,dtype=np.float64,cluster=None,chi=1,inner_like=None,like=None):
    cluster,chi=_get_cluster_chi(shape,cluster,chi)
    if isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        if len(shape)<2:
            raise ValueError("TensorTrainSlice has at least 2 dimensions.")
        else:
            chi=[shape[0]]+list(chi)+[shape[-1]]
            shape=shape[1:-1]
    elif isinstance(like,TensorTrainArray) or like is None:
        chi=[1]+list(chi)+[1]
    else:
        return np.zeros(shape,dtype,like=like)
    if inner_like is not None:
        ms=[np.zeros([c1]+list(s)+[c2],dtype,like=inner_like) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
    else:
        ms=[np.zeros([c1]+list(s)+[c2],dtype) for c1,s,c2 in zip(chi[:-1],cluster,chi[1:])]
        # not ideal, maybe numpy changes that like arg cannot be None
    if like is None:
        return TensorTrainArray.frommatrices(ms)
    if like is TensorTrainSlice:
        return TensorTrainSlice.frommatrices(ms)
    else:
        return like.__class__.frommatrices(ms)

def zeros_slice(shape,dtype=np.float64,cluster=None,chi=1,inner_like=None):
    return zeros(shape,dtype,cluster,chi,inner_like,like=TensorTrainSlice)

@implement_function(np.zeros_like)
def zeros_like(prototype, dtype=None, shape=None, cluster=None, chi=None,inner_like=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    if inner_like is None:
        inner_like=prototype.M[0]
    return zeros(shape,dtype,cluster,chi,inner_like,prototype)

@implement_function(np.ones)
def ones(shape,dtype=np.float64,cluster=None,chi=1,inner_like=None,like=None):
    shape,cluster=_get_shape_cluster_chi(shape,cluster)
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
    ms=[np.ones(s,dtype,order,like=inner_like)/s[0]/s[-1] for s in zip(chi[1:],*cluster,chi[:-1])]
    return like.__class__.from_matrix_list(ms)
@implement_function(np.ones_like)
def ones_like(prototype, dtype=None, shape=None, cluster=None, chi=None,inner_like=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    if inner_like is None:
        inner_like=prototype.M[0]
    return ones(shape,dtype,cluster,chi,inner_like,prototype)

@implement_function(np.full)
def full(shape,fill_value,dtype=None,cluster=None,chi=1,inner_like=None,like=None):
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
    ms=[np.full(s,val,dtype,order,like=inner_like)/s[0]/s[-1] for s in zip(chi[1:],*cluster,chi[:-1])]
    return like.__class__.from_matrix_list(ms)

@implement_function(np.full_like)
def full_like(prototype, fill_value, dtype=None, shape=None, cluster=None, chi=None,inner_like=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    if inner_like is None:
        inner_like=prototype.M[0]
    return full(shape,fill_value,dtype,cluster,chi,inner_like,prototype)

@implement_function(np.eye)
def eye(N, M=None, k=0, dtype=np.float64, cluster=None,chi=1, inner_like=None, like=None):
    if k!=0:
        raise NotImplementedError("Diagonal mps with k!=0 are not yet supported")
    raise NotImplementedError("Not yet")
    # return like.__class__.from_matrix_list(np.eye(inner_like))
@implement_function(np.identity)
def identity(n,dtype=None,cluster=None,chi=1,like=None):
    return eye(N=n,dtype=dtype,cluster=cluster,chi=chi,like=like)

@implement_function(np.array)
def array(ar, dtype=None, cluster=None, copy=True, ndmin=0, like=None):
    '''
        like defaults to TensorTrainArray
    '''
    if isinstance(like,TensorTrainArray) or like is None:
        if isinstance(ar,TensorTrainArray):
            pass
        elif isinstance(ar,TensorTrainSlice):
            #recluster, then recast
            pass
        else:
            ar=ar[None,...,None]
            return like.__class__(array_to_ttslice(ar,find_balanced_cluster(ar.shape),la.qr))
    elif isinstance(like,TensorTrainSlice) or like is TensorTrainSlice:
        return like.__class__(array_to_ttslice(ar,find_balanced_cluster(ar.shape),la.qr))
    else:
        return np.array(ar,dtype=dtype,copy=copy,ndim=ndim,like=like) #fallback to numpy
def slice(ar, dtype=None, cluster=None, copy=True, ndmin=0):
    return array(ar,dtype,cluster,copy,ndim,like=TensorTrainSlice)
@implement_function(np.asarray)
def asarray(ar, dtype=None,cluster=None,like=None):
    array(ar,dtype,cluster=cluster,inner_like=inner_like,like=like,copy=False)

@implement_function(np.asanyarray)
def asanyarray(ar, dtype=None,cluster=None,like=None):
    array(ar,dtype,cluster=cluster,inner_like=inner_like,like=like,copy=False)

@implement_function(np.frombuffer)
def asarray(buffer, dtype=float, count=- 1, offset=0, cluster=None, like=None):
    array(np.frombuffer(buffer,dtype,count,offset),dtype=dtype,cluster=cluster,inner_like=inner_like,like=like,copy=False)
