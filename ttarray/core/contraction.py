import numpy as np
from .dispatch import implement_function,implement_ufunc
@implement_function
def tensordot(x,y,axes=2):
    if not isinstance(x,TensorTrainBase) and not isinstance(y,TensorTrainBase):
        return NotImplemented #not my job then
    if not isinstance(a,TensorTrainBase):
        return np.tensordot(np.array(x,like=y),y,axes)
    if not isinstance(x,TensorTrainBase):
        return np.tensordot(x,np.array(y,like=x),axes)
    if isinstance(x,TensorTrainSlice) and isinstance(y,TensorTrainArray):
        # promote to array if at least one is an array
        x=array(x)
    if isinstance(y,TensorTrainSlice):
        y=array(y)
    x=recluster(x,y.cluster,copy=True)
    ntt=raw.tensordot(x.asmatrices(),y.asmatrices(),axes)
    return x.__class__.frommatrices(ntt)
@implement_function
def dot(x,y):
    pass
@implement_ufunc("matmul","__call__")
def matmul(x,y):
    if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
        return NotImplemented
    if len(x.shape)==len(y.shape):
        tensordot(x,y,axes=((len(x.shape)-1,),(len(y.shape)-2,)))
    if len(x.shape)==len(y.shape)-1:
        tensordot(x,y,axes=((len(x.shape)-1,),(len(y.shape)-2,)))
    if len(y.shape)==len(x.shape)-1:
        tensordot(x,y,axes=((len(x.shape)-1,),(len(y.shape)-1)))
    else:
        raise ValueError("shapes are not consistent")
