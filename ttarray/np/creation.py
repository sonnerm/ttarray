from .. import TensorTrainArray,TensorTrainSlice
import numpy as np
def _get_shape_cluster_chi(shape,cluster,chi):
    if shape is None:
        return None

@implement_function(np.empty)
def empty(shape,dtype=np.float64, cluster=None,chi=1, inner_like=None, like=None):
    '''
        Create an empty TensorTrainSlice or TensorTrainArray
    '''
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
    ms=[np.empty(s,dtype,order,like=inner_like) for s in zip(chi[1:],*cluster,chi[:-1])]
    return like.__class__.from_matrix_list(ms)
@implement_function(np.empty_like)
def empty_like(prototype, dtype=None, shape=None, cluster=None, chi=None,inner_like=None):
    shape,cluster,chi=_get_shape_cluster_chi(shape,cluster,chi)
    if shape is None:
        shape,cluster,chi=prototype.shape,prototype.cluster,prototype.chi
    if dtype is None:
        dtype=prototype.dtype
    if inner_like is None:
        inner_like=prototype.M[0]
    return empty(shape,dtype,cluster,chi,inner_like,prototype)


@implement_function(np.zeros)
def zeros(shape,dtype=np.float64,cluster=None,chi=1,inner_like=None,like=None):
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
    ms=[np.zeros(s,dtype,order,like=inner_like) for s in zip(chi[1:],*cluster,chi[:-1])]
    return like.__class__.from_matrix_list(ms)

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
def array(ar, dtype=None, cluster=None, copy=True, ndmin=0, inner_like=None, like=None):
    mps=[]
    shape,cluster,chi=_get_shape_cluster_chi(ar.shape,cluster,chi)
    car=ar.reshape(ar.shape[0],-1)
    for ds in zip(dims):
        car=car.reshape((car.shape[0]*_product(ds),-1))
        if canonicalize:
            q,r=qr(car)
        else:
            q=np.eye(car.shape[0],like=car)
            r=car
        mps.append(q.reshape((-1,)+ds+(q.shape[1],)))
        car=r
    mps[-1]=np.tensordot(mps[-1],r,axes=((-1,),(0,)))
    return cls(mps)
@implement_function(np.asarray)
def asarray(ar, dtype=None,cluster=None,inner_like=None,like=None):
    array(ar,dtype,cluster,inner_like,like,copy=False)

@implement_function(np.asanyarray)
def asarray(ar, dtype=None,cluster=None,inner_like=None,like=None):
    array(ar,dtype,cluster=cluster,inner_like=inner_like,like=like,copy=False)

@implement_function(np.frombuffer)
def asarray(buffer, dtype=float, count=- 1, offset=0, cluster=None, inner_like=None, like=None):
    array(np.frombuffer(buffer,dtype,count,offset),dtype=dtype,cluster=cluster,inner_like=inner_like,like=like,copy=False)
