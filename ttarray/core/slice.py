from .base import TensorTrainBase
from .. import raw
from .dispatch import HANDLER_UFUNC_SLICE,HANDLER_FUNCTION_SLICE
import numpy.linalg as la
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import functools
def _flatten_ttslice(seq):
    ret=[]
    for s in seq:
        if isinstance(s,TensorTrainSlice):
            ret.extend(s._data)
        ret.append(s)
    return ret
def _product(seq):
    return functools.reduce(lambda a,b:a*b,seq,1) #can't use np.prod since might overflow

def _normalize_axes(axes):
    if axes==None:
        axes=list(range(r))[::-1]
    axes=[a if a>0 else L+a for a in axes]
    return axes
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
    '''
        Represents a TensorTrain with uncontracted boundaries which correspond
        to the first and last dimension.
    '''
    def __init__(self,matrices,center=None):
        '''
            Construct a TensorTrainSlice from a list of matrices and
            orthogonality center. The arguments to the constructor might change,
            use :py:meth:`TensorTrainSlice.frommatrices` or
            :py:meth:`frommatrices_slice` instead. NB: Only checks the
            consistency of bonds, not the canonical property.
        '''
        self._data=matrices
        self._center=center #orthogonality center
        self._check_consistency()
    @property
    def center(self):
        '''
            Orthogonality center of the TensorTrainSlice, ``None`` if unknown or
            TensorTrainSlice not in canonical form. Does not perform any
            calculations and thus runs in O(1)
        '''
        return self._center
    def setcenter_unchecked(self,center):
        '''
            Sets the orthogonality center to ``center`` without checking if the
            TensorTrainSlice is really canonical with this orthogonality center.
            This method is intended for cases where the canonical property is
            ensured
        '''
        self._center=center
    def clearcenter(self):
        '''
            Clears the stored value of the orthogonality center, should be
            invoked after each operation which might violate the canonical
            property.
        '''
        self._center=None
    @property
    def M(self):
        '''
            Gives access to the underlying matrices in a safe way, handles very
            similar to a bare python ``list``, but ensures that invariants are
            upheld at any step
        '''
        return _TensorTrainSliceData(self)
    @property
    def shape(self):
        '''
            Returns the shape of the TensorTrainSlice as a tuple of (python!)
            integers. Since TensorTrains can represent very large vectors the
            entries can easily overflow 64 bit integers.
        '''
        rank=len(self._data[0].shape)
        shape=[self._data[0].shape[0]]
        shape.extend([_product(c[d] for c in self.cluster) for d in range(rank-2)])
        shape.append(self._data[-1].shape[-1])
        return tuple(shape)
    @property
    def cluster(self):
        '''
            the external dimensions of each tensor in the TensorTrainSlice.
        '''
        return tuple(m.shape[1:-1] for m in self._data)
    @property
    def chi(self):
        '''
            the internally contracted (virtual) dimensions of the TensorTrainSlice
        '''
        return tuple(m.shape[0] for m in self._data[1:])
    @property
    def L(self):
        '''
            the number of tensors in the TensorTrainSlice
        '''
        return len(self._data)
    @property
    def dtype(self):
        '''
            the dtype of the TensorTrainSlice
        '''
        return np.result_type(*self._data) #maybe enforce all matrices to the same dtype eventually
    def __repr__(self):
        return "TensorTrainSlice<dtype=%s, shape=%s, L=%s, cluster=%s, chi=%s>"%(self.dtype,self.shape,self.L,self.cluster,self.chi)
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
    def is_canonical(self,center=None):
        '''
            Checks if the TensorTrainSlice is in canonical form with the given
            orthogonality center ``center``. If no center is provided it will
            find the first orthogonality center if it exists and set
            py:property:center accordingly.
            :returns True if self is canonical with orthogonality center
            ``center``, False otherwise

        '''
        if self._center is not None and (center is None or center==self._center):
            return True
        elif center is not None:
            if raw.is_canonical(self.asmatrices_unchecked(),center):
                self._center=center
                return True
            else:
                return False
        else:
            self._center=raw.find_orthogonality_center(self.asmatrices_unchecked())
            return self._center is not None

    def canonicalize(self,center=None,copy=False,qr=la.qr):
        '''
            Canonicalizes the TensorTrainSlice
        '''
        if copy:
            ret=self.copy()
        else:
            ret=self
        if self._center is None:
            raw.canonicalize(ret.asmatrices_unchecked(),center,qr=qr)
            ret._center=center
        else:
            raw.shift_orthogonality_center(ret.asmatrices_unchecked(),self._center,center,qr=qr)
            ret._center=center
        return ret
    def singular_values(self,copy=False,qr=la.qr,svd=la.svd):
        # if copy:
        #     ret=self.copy()
        # else:
        #     ret=self
        # if ret._center is None:
        #     ret.canonicalize(-1,qr=qr)
        # return raw.singular_values(ret.asmatrices_unchecked(),center,svd=svd)
        pass

    @classmethod
    def frommatrices(cls,matrices):
        '''
            Build a TensorTrainSlice from a list of matrices with the internal indices being first and last
        '''
        return cls(matrices)
    def __array__(self,dtype=None):
        '''
            Implementation of the numpy ``__array__`` protocol. See
            :py:func:``ndarray.__array__``
        '''
        return np.asarray(self.todense(),dtype)
    @classmethod
    def fromdense(cls,ar,dtype=None,cluster=None):
        '''
            Convert a dense array into a TensorTrainSlice. Only does copying and
            initializing, O(nitems).
        '''
        if cluster is None:
            cluster=raw.find_balanced_cluster(np.shape(ar)[1:-1])
        if dtype is not None:
            ar=ar.astype(dtype=dtype,copy=False)#change dtype if necessary
        tts=raw.dense_to_ttslice(ar,cluster)
        return cls.frommatrices(tts)
    def todense(self):
        '''
            Convert ``self`` into a dense array of the same kind as the
            underlying data
        '''
        return raw.ttslice_to_dense(self._data)
    def asmatrices(self):
        '''
            Returns a copy of the list of underlying matrices.
        '''
        return [x.copy() for x in self._data]
    def asmatrices_unchecked(self):
        '''
            Doesn't copy, if invariants are violated that is your problem
        '''
        return self._data
    def setmatrices(self,dat):
        bup=self._data
        self._data=[x.copy() for x in self.dat]
        try:
            self._check_consistency()
            self.clearcenter()
        except ValueError as e:
            self._data=bup
            raise e

    def setmatrices_unchecked(self,dat):
        self._data=dat

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
    def transpose(self,*axes):
        r=len(self.shape)
        axes=_normalize_axes(axes)
        if axes[0]!=0 and axes[-1]!=r:
            raise NotImplementedError("transposing of the boundaries of a TensorTrainSlice is not supported yet")
        else:
            return self.__class__.frommatrices([x.transpose(axes) for x in a.M])
    def recluster(self,newcluster=None,copy=False):
        if newcluster is None:
            newcluster = raw.find_balanced_cluster(self.shape)
        if not copy:
            self.setmatrices_unchecked(raw.recluster_ttslice(self.asmatrices_unchecked()))
            return self
        else:
            self.__class__.frommatrices(raw.recluster_ttslice(self.asmatrices()))
    def copy(self):
        return self.__class__.frommatrices([x.copy() for x in self.asmatrices_unchecked()])
