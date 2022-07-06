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
    yc=tuple(tuple(ycc[i] for i in axes[1]) for ycc in y.cluster)
    x=recluster(x,yc,axes=axes[0],copy=True)
    ntt=raw.tensordot(x.tomatrices(),y.tomatrices(),axes)
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

# @implement_function
# def einsum(a,out=None):
#     if not isinstance(x,TensorTrainBase) and isinstance(y,TensorTrainBase):
#         return NotImplemented
#     if len(x.shape)==1 and len(y.shape)==1:
#         return tensordot(x.conj(),y,axes=((0,),(0,)),out=out)
#     else:
#         raise ValueError("shapes are not consistent")
@implement_function()
def trace(a,offset=0, axis1=0,axis2=1,dtype=None,out=None):
    if offset!=0:
        raise NotImplementedError("off diagonal trace not yet implemented")
    if out!=None:
        raise NotImplementedError("out argument not yet implemented")
    if axis1<0:
        axis1=len(a.shape)+axis1
    if axis2<0:
        axis2=len(a.shape)+axis2
    if axis1>=len(a.shape) or axis1<0:
        raise ValueError("axis1 out of bounds: %i (rank: %i)"%(axis1,len(a.shape)))
    if axis2>=len(a.shape) or axis2<0:
        raise ValueError("axis2 out of bounds: %i (rank: %i)"%(axis2,len(a.shape)))
    if isinstance(a,TensorTrainSlice):
        if axis1==0 or axis1==len(a.shape)-1 or axis2==0 or axis2==len(a.shape)-1:
            raise NotImplementedError("Tracing the boundary is not yet supported for TensorTrainSlices")
        return TensorTrainSlice.frommatrices_unchecked([np.trace(x,0,axis1+1,axis2+1,dtype) for x in a.tomatrices_unchecked()])
    elif isinstance(a,TensorTrainArray):
        return TensorTrainArray.frommatrices_unchecked([np.trace(x,0,axis1+1,axis2+1,dtype) for x in a.tomatrices_unchecked()])
    else:
        return NotImplemented
