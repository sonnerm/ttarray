import numpy as np
from . import full,array,recluster
@implement_ufunc("add","__call__")
def add(x,y):
    if np.isscalar(x) and np.isscalar(y):
        return NotImplemented
    if np.isscalar(x):
        #can't use np.full since we want to pass cluster
        if isinstance(y,TensorTrainArray):
            x=full(y.shape,x,cluster=y.cluster)
        elif isinstance(y,TensorTrainSlice):
            x=full_slice(y.shape,x,cluster=y.cluster)
        else:
            return NotImplemented
    if np.isscalar(y):
        if isinstance(x,TensorTrainArray):
            y=full(x.shape,y,cluster=x.cluster)
        elif isinstance(x,TensorTrainSlice):
            y=full_slice(x.shape,y,cluster=x.cluster)
        else:
            return NotImplemented
    if not isinstance(x,TensorTrainBase) and not isinstance(y,TensorTrainBase):
        return NotImplemented #not my job then
    if not isinstance(y,TensorTrainBase):
        return np.add(np.array(x,like=y),y)
    if not isinstance(x,TensorTrainBase):
        return np.add(x,np.array(y,like=x))

    if isinstance(x,TensorTrainSlice) and isinstance(y,TensorTrainArray):
        # promote to array if at least one is an array
        x=array(x)
    if isinstance(y,TensorTrainSlice):
        y=array(y)
    x=recluster(x,y.cluster)
    ntt=add_ttslice(x.asmatrices(),y.asmatrices())
    return x.__class__.frommatrices(ntt)
@implement_ufunc("multiply","__call__")
def multiply(x,y):
    pass

@implement_ufunc("matmul","__call__")
def matmul(x,y):
    pass
