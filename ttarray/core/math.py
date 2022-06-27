import numpy as np
import ttarray.raw as raw
from .creation import full,array
from .reshape import recluster
from .dispatch import implement_ufunc
from .base import TensorTrainBase
from .array import TensorTrainArray
from .slice import TensorTrainSlice
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
    x=recluster(x,y.cluster,copy=True)
    ntt=raw.add_ttslice(x.tomatrices(),y.tomatrices())
    return x.__class__.frommatrices(ntt)
@implement_ufunc("multiply","__call__")
def multiply(x,y):
    if np.isscalar(x) and np.isscalar(y):
        return NotImplemented
    if np.isscalar(x):
        mats=y.tomatrices()
        mats[0]*=x
        return y.__class__.frommatrices(mats)
    if np.isscalar(y):
        mats=x.tomatrices()
        mats[0]*=y
        return x.__class__.frommatrices(mats)
    if not isinstance(x,TensorTrainBase) and not isinstance(y,TensorTrainBase):
        return NotImplemented #not my job then
    if not isinstance(y,TensorTrainBase):
        return np.multiply(np.array(x,like=y),y)
    if not isinstance(x,TensorTrainBase):
        return np.multiply(x,np.array(y,like=x))

    if isinstance(x,TensorTrainSlice) and isinstance(y,TensorTrainArray):
        # promote to array if at least one is an array
        x=array(x)
    if isinstance(y,TensorTrainSlice) and isinstance(x,TensorTrainArray):
        y=array(y)
    x=recluster(x,y.cluster,copy=True)
    ntt=raw.multiply_ttslice(x.tomatrices(),y.tomatrices())
    return x.__class__.frommatrices(ntt)

@implement_ufunc("conjugate","__call__")
def conjugate(x):
    if not isinstance(x,TensorTrainBase):
        return NotImplemented
    return x.conj()
conj=conjugate
