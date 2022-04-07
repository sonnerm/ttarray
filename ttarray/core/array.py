from numpy.lib.mixins import NDArrayOperatorsMixin
import numpy as np
import numpy.linalg as la
from .slice import TensorTrainSlice,_TensorTrainSliceData,_normalize_axes
from .base import TensorTrainBase
from .dispatch import HANDLER_UFUNC_ARRAY,HANDLER_FUNCTION_ARRAY
from .. import raw
class _TensorTrainArrayData(_TensorTrainSliceData):
    def __setitem__(self,ind,it):
        super().__setitem__(ind,it)
        if self._tts._data[0].shape[0]!=1 or self._tts._data[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")

    def __delitem__(self,ind):
        super().__delitem__(ind)
        if self._tts._data[0].shape[0]!=1 or self._tts._data[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")

class TensorTrainArray(TensorTrainBase,NDArrayOperatorsMixin):
    '''
        TensorTrainArray represents a ndimensional array in tensor train format.

        For performance reasons, all data is assumed to be *owned* by the
        TensorTrainArray instance. All functions, except those which end in
        *_unchecked will *copy* data if appropriate.
    '''
    def __init__(self,tts):
        '''
           Creates a TensorTrainArray from a TensorTrainSlice where the boundary dimensions are both 1.

           This constructor is **not** part of the stable API and might change
           in the future. Do **not** use it in downstream projects. Instead use
           :py:meth:`TensorTrainArray.fromslice` or
           :py:meth:`TensorTrainArray.fromslice_unchecked` as appropriate
        '''
        if tts._data[0].shape[0]!=1 or tts._data[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
        self._tts=tts
    @property
    def M(self):
        return _TensorTrainArrayData(self._tts)
    @property
    def shape(self):
        return self._tts.shape[1:-1]
    @property
    def cluster(self):
        return self._tts.cluster
    @property
    def chi(self):
        return self._tts.chi
    @property
    def dtype(self):
        return self._tts.dtype
    @property
    def L(self):
        return self._tts.L
    @property
    def center(self):
        return self._tts.center

    def __repr__(self):
        return "TensorTrainArray<dtype=%s, shape=%s, L=%s, cluster=%s, chi=%s>"%(self.dtype,self.shape,self.L,self.cluster,self.chi)
    def __array__(self,dtype=None):
        return np.asarray(self.todense(),dtype)
    @classmethod
    def fromdense(cls,ar,dtype=None,cluster=None):
        return cls(TensorTrainSlice.fromdense(ar[None,...,None],dtype,cluster))
    def todense(self):
        return self._tts.todense()[0,...,0]
    @classmethod
    def frommatrices(cls,mpl):
        return cls(TensorTrainSlice.frommatrices(mpl))
    @classmethod
    def frommatrices_unchecked(cls,mpl):
        return cls(TensorTrainSlice.frommatrices_unchecked(mpl))
    @classmethod
    def fromslice(self,tts):
        return cls(tts.copy())
    @classmethod
    def fromslice_unchecked(self,tts):
        return cls(tts)
    def asslice(self):
        return self._tts.copy()
    def asslice_unchecked(self):
        return self._tts
    def asmatrices(self):
        return self._tts.asmatrices() #already does copying
    def asmatrices_unchecked(self):
        return self._tts.asmatrices_unchecked()
    def __array_function__(self,func,types,args,kwargs):
        f=HANDLER_FUNCTION_ARRAY.get(func.__name__,None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        f=HANDLER_UFUNC_ARRAY.get((ufunc,method),None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)

    def transpose(self,*axes):
        r=len(self.shape)
        axes=_normalize_axes(axes)
        naxes=[0]+axes+[r]
        return self.__class__.frommatrices([x.transpose(naxes) for x in a.M])
    def recluster(self,newcluster=None,copy=False):
        tts=self._tts.recluster(newcluster,copy)
        if not copy:
            return self
        else:
            self.__class__.fromslice(tts)
    def copy(self):
        return self.fromslice(self._tts.copy())

    def setmatrices(self,dat):
        if dat[0].shape[0]!=1 or dat[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
        self._tts.setmatrices(dat)

    def setmatrices_unchecked(self,dat):
        self._tts.setmatrices_unchecked(dat)
    def canonicalize(self):
        return self._tts.canonicalize()

    def is_canonical(self):
        return self._tts.is_canonical()
