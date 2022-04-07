import numpy as np
from .dispatch import implement_function,implement_ufunc
@implement_function
def tensordot(a,b,axes=2):
    if isinstance(a,TensorTrainArray):
        if isinstance(b,TensorTrainArray):
            a=recluster(a,b.cluster)
            return raw.tensordot(a,b)
        else:#other array like
            pass
    else:
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
