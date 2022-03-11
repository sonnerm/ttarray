import numpy as np
import copy
from functools import reduce
from .raw import ttslice_to_dense,dense_to_ttslice,trivial_decomposition,find_balanced_cluster
from numpy.lib.mixins import NDArrayOperatorsMixin
HANDLER_FUNCTION_ARRAY={}
HANDLER_UFUNC_ARRAY={}
HANDLER_FUNCTION_SLICE={}
HANDLER_UFUNC_SLICE={}
def implement_function(func=None,selector=None):
    def implement_function_decorator(f):
        if func is None:
            fname=f.__name__
        else:
            fname=func
        if selector=="array" or selector is None:
            HANDLER_FUNCTION_ARRAY[fname]=f
        if selector=="slice" or selector is None:
            HANDLER_FUNCTION_SLICE[fname]=f
        return f
    return implement_function_decorator
def implement_ufunc(ufunc,method,selector=None):
    def implement_ufunc_decorator(f):
        if selector=="array" or selector is None:
            HANDLER_UFUNC_ARRAY[(func,method)]=f
        if selector=="slice" or selector is None:
            HANDLER_UFUNC_SLICE[(func,method)]=f
        return f
    return implement_ufunc_decorator


def _product(seq):
    return reduce(lambda a,b:a*b,seq,1)
def _flatten_ttslice(seq):
    ret=[]
    for s in seq:
        if isinstance(s,TensorTrainSlice):
            ret.extend(s._data)
        ret.append(s)
    return ret

class TensorTrainBase:
    # So far this is just for isinstance
    pass

class _TensorTrainSliceData:
    def __init__(self,tts):
        self._tts=tts
    def __getitem__(self,ind):
        if isinstance(ind,slice):
            if ind.step==1:
                return self._tts.__class__(self._tts._data[ind])
            else:
                raise IndexError("Slicing with a step not equal to one not supported for TensorTrains")
        else: #single index i guess
            return self._tts._data[ind]

    def __len__(self):
        return len(self._data)
    def __setitem__(self,ind,it):
        it=_flatten_ttslice(it)
        if isinstance(ind,slice):
            if ind.step==1:
                self._tts._data[ind]=it
            else:
                raise IndexError("Slicing with a step not equal to one not supported for TensorTrains")
        else: #single index i guess
            self._tts._data[ind]=it
        self._check_consistency()
    def __delitem__(self,ind):
        del self._tts._data[ind]
        self._check_consistency()
    def __iter__(self):
        return self._tts._data.__iter__()

class TensorTrainSlice(TensorTrainBase,NDArrayOperatorsMixin):
    def __init__(self,matrices):
        self._data=matrices
        self._check_consistency()
    @property
    def M(self):
        return _TensorTrainSliceData(self)
    @property
    def shape(self):
        rank=len(self._data[0].shape)
        shape=[self._data[0].shape[0]]
        shape.extend([_product(c[d] for c in self.cluster) for d in range(rank-2)])
        shape.append(self._data[-1].shape[-1])
        return tuple(shape)
    @property
    def cluster(self):
        return tuple(m.shape[1:-1] for m in self._data)
    @property
    def chi(self):
        return tuple(m.shape[0] for m in self._data[1:])
    @property
    def L(self):
        return len(self._data)
    @property
    def dtype(self):
        return np.result_type(*self._data) #maybe enforce all matrices to the same dtype eventually

    def _check_consistency(self):
        if self.L<1:
            raise ValueError("There must be at least one matrix in the TensorTrainSlice")
        rank=len(self._data[0].shape)
        nd=self._data[0].shape[0]
        for m in self._data:
            if len(m.shape)!=rank:
                raise ValueError("All matrices in a TensorTrainSlice need to have the same rank")
            if m.shape[0]!=nd:
                raise ValueError("The virtual bonds need to be consistent")
            nd=m.shape[-1]
    @classmethod
    def frommatrices(cls,matrices):
        return cls(matrices)
    def __array__(self,dtype=None):
        return np.asarray(self.todense(),dtype)
    @classmethod
    def fromdense(cls,ar,dtype=None,cluster=None):
        if cluster is None:
            cluster=find_balanced_cluster(np.shape(ar)[1:-1])
        if dtype is not None:
            ar=ar.astype(dtype=dtype,copy=False)#change dtype if necessary
        tts=dense_to_ttslice(ar,cluster,trivial_decomposition)
        return cls.frommatrices(tts)
    def todense(self):
        return ttslice_to_dense(self._data)
    def asmatrices(self):
        '''
            This method is called 'as_matrix_list' since it returns a view
        '''
        return list(self._data) #shallow copy to protect invariants
    def __array_function__(self,func,types,args,kwargs):
        f=HANDLER_FUNCTION_SLICE.get(func.__name__,None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        f=HANDLER_UFUNC_SLICE.get((ufunc,method),None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)

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
    def __init__(self,tts):
        self._tts=tts
        if tts._data[0].shape[0]!=1 or tts._data[-1].shape[-1]!=1:
            raise ValueError("TensorTrainArrays cannot have a non-contracted boundary")
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

    def __array__(self,dtype=None):
        return np.asarray(self.todense(),dtype)
    # @classmethod
    # def __array_wrap__(cls,arr,context=None):
    #     return cls.fromdense(arr)
    @classmethod
    def fromdense(cls,ar,dtype=None,cluster=None):
        return cls(TensorTrainSlice.fromdense(ar[None,...,None],dtype,cluster))
    def todense(self):
        return self._tts.todense()[0,...,0]
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
        f=HANDLER_FUNCTION_ARRAY.get(func.__name__,None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        f=HANDLER_UFUNC_ARRAY.get((ufunc,method),None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
