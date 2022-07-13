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
    def _check_consistency(self):
        if self._tts.shape[0]!=1 or self._tts.shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
        self._tts._check_consistency()
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
    @property
    def T(self):
        return self.transpose()
    def clearcenter(self):
        self._tts.clearcenter()
    def setcenter_unchecked(self,ncenter):
        self._tts.setcenter_unchecked(ncenter)
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
    def fromslice(cls,tts):
        return cls(tts.copy())
    @classmethod
    def fromslice_unchecked(cls,tts):
        return cls(tts)
    def toslice(self):
        return self._tts.copy()
    def toslice_unchecked(self):
        return self._tts
    def tomatrices(self):
        return self._tts.tomatrices() #already does copying
    def tomatrices_unchecked(self):
        return self._tts.tomatrices_unchecked()
    def __array_function__(self,func,types,args,kwargs):
        f=HANDLER_FUNCTION_ARRAY.get(func.__name__,None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def __array_ufunc__(self,ufunc,method,*args,**kwargs):
        f=HANDLER_UFUNC_ARRAY.get((ufunc.__name__,method),None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)

    def transpose(self,*axes):
        r=len(self.shape)
        axes=_normalize_axes(r,axes)
        naxes=[0]+[a+1 for a in axes]+[r+1]
        return self.__class__.frommatrices_unchecked([x.transpose(naxes) for x in self.M])
    def recluster(self,newcluster=None,axes=None,copy=False):
        if axes is not None:
            axes=tuple(a+1 for a in axes)
        tts=self._tts.recluster(newcluster,axes=axes,copy=copy)
        if not copy:
            return self
        else:
            return self.__class__.fromslice(tts)
    def copy(self):
        return self.fromslice(self._tts.copy())

    def setmatrices(self,dat):
        if dat[0].shape[0]!=1 or dat[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
        self._tts.setmatrices(dat)

    def setmatrices_unchecked(self,dat):
        self._tts.setmatrices_unchecked(dat)
    def canonicalize(self,center=None,copy=False,qr=la.qr):
        return self.__class__.fromslice_unchecked(self._tts.canonicalize(center,copy,qr))
    def truncate(self,chi_max=None,cutoff=0.0,left=0,right=-1,qr=la.qr,svd=la.svd):
        return self._tts.truncate(chi_max,cutoff,left,right,qr,svd)
    def singular_values(self,left=0,right=-1,ro=False,svd=la.svd,qr=la.qr):
        return self._tts.singular_values(left,right,svd,qr)

    def is_canonical(self,center=None):
        return self._tts.is_canonical(center)
    def conj(self):
        return self.__class__.fromslice_unchecked(self._tts.conj())
