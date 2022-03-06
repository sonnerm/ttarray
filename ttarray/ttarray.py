import numpy as np
import copy
from functools import reduce
from .dispatch import dispatch_array_ufunc,dispatch_array_function
from .raw import ttslice_to_array,array_to_ttslice
import numpy.linalg as la
from numpy.lib.mixins import NDArrayOperatorsMixin
def _product(seq):
    return reduce(lambda a,b:a*b,seq,1)
def _flatten_mps(mpsl):
    pass
class TensorTrainBase:
    # So far this is just for isinstance
    pass

class _TensorTrainSliceData:
    def __init__(self,mps):
        self.mps=mps
    def __getitem__(self,ind):
        if isinstance(ind,slice):
            if ind.step==1:
                return self.mps.__class__(self.mps._data[ind])
            else:
                raise IndexError("Slicing with a step not equal to one not supported for TensorTrains")
        else: #single index i guess
            return self.mps._data[ind]

    def __len__(self):
        return len(self._data)
    def __setitem__(self,ind,it):
        it=_flatten_mps(it)
        if isinstance(ind,slice):
            if ind.step==1:
                self.mps._data[ind]=it
            else:
                raise IndexError("Slicing with a step not equal to one not supported for TensorTrains")
        else: #single index i guess
            self.mps._data[ind]=it
        self._update_attributes()
    def __delitem__(self,ind):
        del self.mps._data[ind]
        self._update_attributes()
    def __iter__(self):
        return self.mps._data.__iter__()

class TensorTrainSlice(TensorTrainBase,NDArrayOperatorsMixin):
    def __init__(self,matrices):
        self._data=matrices
        self.M=_TensorTrainSliceData(self)
        self._update_attributes()
    def _update_attributes(self):
        self.L=len(self._data)
        if self.L<1:
            raise ValueError("There must be at least one matrix in the TensorTrainSlice")
        self.chi=tuple(m.shape[0] for m in self._data[1:])
        self.cluster=tuple(m.shape[1:-1] for m in self._data)
        rank=len(self._data[0].shape)
        for m in self._data:
            if len(m.shape)!=rank:
                raise ValueError("All matrices in a TensorTrainSlice need to have the same rank")
        shape=[self._data[0].shape[0]]
        shape.extend([_product(c[d] for c in self.cluster) for d in range(rank-2)])
        shape.append(self._data[-1].shape[-1])
        self.shape=tuple(shape)
        self.dtype=np.common_type(*self._data)
    @classmethod
    def frommatrices(cls,matrices):
        cls(matrices)
    def __array__(self,dtype=None):
        return np.asarray(self.toarray(),dtype)
    @classmethod
    def fromarray(cls,ar,dims=None,canonicalize=True,qr=la.qr):
        return cls.frommatrices(array_to_ttslice(ar,cluster,qr))
    def toarray(self):
        return ttslice_to_array(self._data)
    def asmatrices(self):
        '''
            This method is called 'as_matrix_list' since it returns a view
        '''
        return list(self._data) #shallow copy to protect invariants
    def __array_function__(self,func,types,args,kwargs):
        return dispatch_array_function(func,types,args,kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        return dispatch_array_ufunc(ufunc,method,args,kwargs)

class _TensorTrainArrayData:
    def __init__(self,tta):
        self._tta=tta
        self._M=tta._tts.M
    def __getitem__(self,ind):
        return self._M.__getitem__(self,ind)
    def __len__(self):
        return self._M.__len__(self)
    def __setitem__(self,ind,it):
        self._M.__setitem__(ind,it)
        self._tta._update_attributes()
    def __delitem__(self,ind):
        self._M.__delitem__(ind)
        self._tta._update_attributes()
    def __iter__(self):
        return self._M.__iter__()
class TensorTrainArray(TensorTrainBase,NDArrayOperatorsMixin):
    def __init__(self,tts):
        self._tts=tts
        self.M=_TensorTrainArrayData(self)
        self._update_attributes()
    def _update_attributes(self):
        if self._tts.shape[0]!=1 or self._tts.shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
        self.shape=self._tts.shape[1:-1]
        self.cluster=self._tts.cluster
        self.dtype=self._tts.dtype
        self.L=self._tts.L
        self.chi=self._tts.chi
    def __array__(self,dtype=None):
        return np.asarray(self.toarray(),dtype)
    @classmethod
    def fromarray(cls,ar,cluster=None):
        return cls(TensorTrainSlice.fromarray(ar,dims,canonicalize))
    def toarray(self):
        return self._tts.toarray()[0,...,0]
    @classmethod
    def frommatrices(cls,mpl):
        return cls(TensorTrainSlice(mpl))
    @classmethod
    def fromslice(self,tts):
        return cls(mps)
    def asslice(self):
        return copy.copy(self._tts) #shallow copy necessary to protect invariants, stilll a view
    def asmatrices(self):
        return self._tts.asmatrices() #already does shallow copying
    def __array_function__(self,func,types,args,kwargs):
        return _dispatch_array_function(func,types,args,kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        return _dispatch_array_ufunc(ufunc,method,args,kwargs)
