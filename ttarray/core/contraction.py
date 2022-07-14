import numpy as np
from .creation import asarray
from .dispatch import implement_function,implement_ufunc
from .base import TensorTrainBase
from .slice import TensorTrainSlice
from .array import TensorTrainArray
from .reshape import recluster
import ttarray.raw as raw
def tensordot_slice(x,y,axes=2,out=None):
    '''
        tensordot, but with slice-aware ordering of uncontracted indices
    '''
    pass
def _format_axes(axes,xr,yr):
    if isinstance(axes,(int,np.integer)):
        axes=(tuple(range(-axes,0)),tuple(range(0,axes)))
    elif isinstance(axes[0],(int,np.integer)):
        axes=((axes[0],),(axes[1],))
    axes=(tuple(c if c>=0 else c+xr for c in axes[0]),tuple(c if c>=0 else c+yr for c in axes[1]))
    return axes
@implement_function()
def tensordot(x,y,axes=2,out=None):
    if out is not None:
        raise NotImplementedError("out is not yet supported")
    axes=_format_axes(axes,len(x.shape),len(y.shape))
    for a1,a2 in zip(axes[0],axes[1]):
        if x.shape[a1]!=y.shape[a2]:
            raise ValueError("Arrays have incompatible dimensions x.shape[%i]=%i and y.shape[%i]=%i"%(a1,x.shape[a1],a2,y.shape[a2]))
    y=asarray(y)
    x=asarray(x)
    yc=tuple(tuple(ycc[i] for i in axes[1]) for ycc in y.cluster)
    x=recluster(x,yc,axes=axes[0],copy=True)
    ntt=raw.tensordot(x.tomatrices(),y.tomatrices(),axes)
    return x.__class__.frommatrices(ntt)
@implement_function()
def dot(x,y,out=None):
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x,y,axes=((0,),(0,)),out=out)
    else:
        return tensordot(x,y,axes=((len(x.shape)-1,),(len(y.shape)-2,)))
@implement_function()
def vdot(x,y,out=None):
    if len(x.shape)==1 and len(y.shape)==1:
        return tensordot(x.conj(),y,axes=((0,),(0,)),out=out)
    else:
        return tensordot(x.conj(),y,axes=((len(x.shape)-1,),(len(y.shape)-2,)),out=out)

@implement_ufunc("matmul","__call__")
def matmul(x,y):
    if len(y.shape)==0 or len(x.shape)==0:
        raise ValueError("Scalar multiplication with matmul is not allowed!")
    if len(y.shape)!=1 and len(y.shape)!=2:
        raise NotImplementedError("shapes other then 1-D and 2-D are not yet supported")
    if len(x.shape)==1:
        return tensordot(x,y,axes=((0,),(0,)))
    elif len(x.shape)==2:
        return tensordot(x,y,axes=((1,),(0,)))
    else:
        raise NotImplementedError("shapes other then 1-D and 2-D are not yet supported")
@implement_function()
def multi_dot(arrays,*,out=None):
    if out is not None:
        raise NotImplementedError("out is not yet supported")
    narrays=[]
    cl=None
    for a in arrays[::-1]:
        a=array(a)
        if cl==-1:
            raise ValueError("Only the first and last array can be 1-d!")
        if cl is not None and len(a.shape)==2:
            a.recluster(a,cl,axes=(1,),copy=False)
        elif cl is not None and len(a.shape)==1:
            a.recluster(a,cl,axes=(0,),copy=False)
            cl=-1
        elif len(a.shape)!=1 and len(a.shape)!=2:
            raise ValueError("All arrays must be 1-d or 2-d")
        cl=[(c[0],) for c in a.cluster]
        narrays.append(a)
    return frommatrices(raw.multi_dot([a.tomatrices_unchecked() for a in narrays[::-1]]))
@implement_function()
def inner(a,b,/):
    a=asarray(a)
    b=asarray(b)
    if len(a.shape)==0 or len(b.shape)==0:
        return a*b
    return tensordot(a,b,axes=((-1,),(-1,)))

# @implement_function
# def einsum(subscripts, *operands, out=None, dtype=None, casting='safe', optimize=False):
#     if out is not None:
#         raise NotImplementedError("out is not yet supported")
#     noperands=[]
#     cld={}

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
            return trace(array(a),offset,axis1,axis2,dtype,out)
        a=a.recluster([(x[axis2],x[axis2]) for x in a.cluster],[axis1,axis2])
        return TensorTrainSlice.frommatrices_unchecked([np.trace(x,0,axis1+1,axis2+1,dtype) for x in a.tomatrices_unchecked()])
    elif isinstance(a,TensorTrainArray):
        a=a.recluster([(x[axis2],x[axis2]) for x in a.cluster],[axis1,axis2])
        return TensorTrainArray.frommatrices_unchecked([np.trace(x,0,axis1+1,axis2+1,dtype) for x in a.tomatrices_unchecked()])
    else:
        return np.trace(a,offset,axis1,axis2,dtype,out)
