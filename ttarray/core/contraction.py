import numpy as np
from .dispatch import implement_function,implement_ufunc
from .base import TensorTrainBase
from .slice import TensorTrainSlice
from .array import TensorTrainArray
from .reshape import recluster
import ttarray.raw as raw
@implement_function()
def tensordot(x,y,axes=2,out=None):
    if isinstance(axes,(int,np.integer)):
        axes=(tuple(range(-axes,0)),tuple(range(0,axes)))
    elif isinstance(axes[0],(int,np.integer)):
        axes=((axes[0],),(axes[1],))
    axes=(tuple(c if c>=0 else c+len(x.shape) for c in axes[0]),tuple(c if c>=0 else c+len(y.shape) for c in axes[1]))
    if not isinstance(x,TensorTrainBase) and not isinstance(y,TensorTrainBase):
        return NotImplemented #not my job then
    if not isinstance(x,TensorTrainBase):
        return np.tensordot(np.array(x,like=y),y,axes)
    if not isinstance(y,TensorTrainBase):
        return np.tensordot(x,np.array(y,like=x),axes)

    if isinstance(x,TensorTrainSlice) and isinstance(y,TensorTrainSlice):
        raise NotImplementedException("tensordot operations between two TensorTrainSlice are not supported yet.")
    elif isinstance(x,TensorTrainSlice):
        # promote to array if at least one is an array
        x=array(x)
    if isinstance(y,TensorTrainSlice):
        y=array(y)
    x=recluster(x,y.cluster,copy=True)
    ntt=raw.tensordot(x.asmatrices(),y.asmatrices(),axes)
    return x.__class__.frommatrices(ntt)
@implement_function()
def dot(x,y,out=None):
    if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
        return NotImplemented
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x,y,axes=((0,),(0,)),out=out)
    else:
        raise ValueError("shapes are not consistent")
@implement_function()
def vdot(x,y,out=None):
    if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
        return NotImplemented
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x.conj(),y,axes=((0,),(0,)),out=out)
    else:
        raise ValueError("shapes are not consistent")


@implement_ufunc("matmul","__call__")
def matmul(x,y):
    if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
        return NotImplemented
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x,y,axes=((0,),(0,)))
    else:
        return tensordot(x,y,axes=((len(x.shape)-1,),(len(y.shape)-2,)))

@implement_function
def einsum(a,out=None):
    if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
        return NotImplemented
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x.conj(),y,axes=((0,),(0,)),out=out)
    else:
        raise ValueError("shapes are not consistent")
