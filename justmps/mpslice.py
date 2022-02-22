import numpy as np
import copy
from functools import reduce
from .dispatch import _dispatch_array_ufunc
from .dispatch import _dispatch_array_function
import numpy.linalg as la
from numpy.lib.mixins import NDArrayOperatorsMixin
PRIMES=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
#First 100 primes, should be way more then sufficient for basically anything
def _decompose_dim(d):
    ls=[]
    for p in PRIMES:
        while d%p==0:
            ls.append(p)
            d=d//p
    if d!=1:
        raise NotImplementedError("Need more Primes to do this")
    return ls[::-1]
def _product(seq):
    return reduce(lambda a,b:a*b,seq,1)
def _calc_shape(ms):
    rank=len(ms[0].shape)
    for m in ms:
        if len(m.shape!=rank):
            raise ValueError("All matrices in a MatrixProductSlice need to have the same rank")
    return (ms[0].shape[0],)+tuple(_product(m.shape[d] for m in ms) for d in range(1,rank-1))+(ms[-1].shape[-1],)
def _flatten_mps(mpsl):
    pass

class _MatrixProductSliceData:
    def __init__(self,mps):
        self.mps=mps
    def __getitem__(self,ind):
        if isinstance(ind,slice):
            if ind.step==1:
                return self.mps.__class__(self.mps._data[ind])
            else:
                raise IndexError("Slicing with a step not equal to one not supported")
        else: #single index i guess
            return self.mps._data[ind]

    def __len__(self):
        return len(self._data)
    def __setitem__(self,ind,it):
        it=_flatten_mps(it)
        if isinstance(ind,slice):
            if ind.step==1:
                return self.mps._data[ind]=it
            else:
                raise IndexError("Slicing with a step not equal to one not supported")
        else: #single index i guess
            return self.mps._data[ind]=it
        self.mps.shape=_calc_shape(matrices) #need to recalculate shape
        self.mps.dtype=np.common_type(matrices) #maybe dtype has changed
    def __delitem__(self,ind):
        del self.mps._data[ind]
        self.mps.shape=_calc_shape(matrices)
        self.mps.dtype=np.common_type(matrices)
    def __iter__(self):
        return self.mps._data.__iter__()

class MatrixProductSlice(NDArrayOperatorsMixin):
    def __init__(self,matrices):
        self._data=matrices
        self.M=_MatrixProductSliceData(self)
        self.shape=_calc_shape(matrices)
        self.dtype=np.common_type(matrices)
    def __array__(self,dtype=None):
        ret=self._data[0]
        for m in self._data[1:]:
            ret=np.tensordot(ret,m,axes=((-1),(0)))
        return np.asarray(ret,dtype)
    @classmethod
    def from_matrix_list(cls,matrices):
        cls(matrices)
    @classmethod
    def from_array(cls,ar,dims=None,canonicalize=True,qr=la.qr):
        mps=[]
        if dims==None:
            dims=[_decompose_dim(s) for s in ar.shape[1:-1]]
        md=max((len(d) for d in dims))
        for da,ad in zip(dims,ar.shape[1:-1]):
            if _product(da)!=ad:
                raise ValueError("Dimensions do not match, needed %i got %i"%(ad,_product(da)))
            if len(da)<md:
                da.extend([1]*(md-len(da)))
        car=ar.reshape(ar.shape[0],-1)
        for ds in zip(dims):
            car=car.reshape((car.shape[0]*_product(ds),-1))
            if canonicalize:
                q,r=qr(car)
            else:
                q=np.eye(car.shape[0],like=car)
                r=car
            mps.append(q.reshape((-1,)+ds+(q.shape[1],)))
            car=r
        mps[-1]=np.tensordot(mps[-1],r,axes=((-1,),(0,)))
        return cls(mps)
    def truncate(self,**kwargs):
        return self
    def as_matrix_list(self):
        return list(self._data) #shallow copy to protect invariants
    def __array_function__(self,func,types,args,kwargs):
        return _dispatch_array_function(func,types,args,kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        return _dispatch_array_ufunc(ufunc,method,args,kwargs)
