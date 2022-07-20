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

def _normalize_axes(r,axes):
    if axes==():
        axes=list(range(r))[::-1]
    axes=[a if a>=0 else r+a for a in axes]
    return axes
def _check_cluster_axes(c1,c2,axes):
    c1=np.array(c1)
    c2=np.array(c2)
    if c1.shape!=c2.shape:
        return False
    return (c1[:,np.array(axes)-1]==c2).all()
class _TensorTrainSliceData:
    def __init__(self,tts):
        self._tts=tts
    def __getitem__(self,ind):
        if isinstance(ind,slice):
            if ind.step==1 or ind.step==None:
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
            if ind.step==1 or ind.step==None:
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
        Represents a array in tensor train format with uncontracted boundaries.
        The first and last dimension correspond to the open boundaries. For
        performance reasons, all data is assumed to be *owned* by the
        TensorTrainSlice instance. To ensure this, all functions, except those
        which end in *_unchecked will *copy* data if appropriate.
    '''
    def __init__(self,data,center=None):
        '''
            Construct a TensorTrainSlice from a list of matrices and
            orthogonality center.

            It is strongly advised to **not** use this constructor in downstream
            projects, since its arguments and semantics might change between
            releases.
            `data` is not copied and can violate invariants if modified outside
            of the TensorTrainSlice class.
        '''
        self._data=data
        self._center=center #orthogonality center
        self._check_consistency()
    @property
    def center(self):
        '''
            Orthogonality center of the TensorTrainSlice, ``None`` if unknown or
            TensorTrainSlice not in canonical form. Does not perform any
            calculations.
        '''
        return self._center
    def setcenter_unchecked(self,center):
        '''
            Sets the orthogonality center to ``center`` without checking if the
            TensorTrainSlice is really canonical with this orthogonality center.
            This method is intended for cases where the (partial) canonical property
            is ensured by conditions outside of this class.

            See also:
            is_canonical, canonicalize

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
            tuple(int): shape of the TensorTrainSlice as tuple of python integers.

            Note
            -----
            tensor trains can represent very high-dimensional arrays, the
            dimensions do not necessarily fit in a 64 bit integer!
        '''
        rank=len(self._data[0].shape)
        shape=[self._data[0].shape[0]]
        shape.extend([_product(c[d] for c in self.cluster) for d in range(rank-2)])
        shape.append(self._data[-1].shape[-1])
        return tuple(shape)
    @property
    def cluster(self):
        ''' tuple(tuple(int)): the external dimensions of each tensor in the TensorTrainSlice. '''
        return tuple(m.shape[1:-1] for m in self._data)
    @property
    def chi(self):
        ''' int: the internal ('virtual') dimensions of the TensorTrainSlice'''
        return tuple(m.shape[0] for m in self._data[1:])
    @property
    def L(self):
        '''
            int: the number of tensors in the TensorTrainSlice
        '''
        return len(self._data)
    @property
    def dtype(self):
        '''
            numpy.dtype: the dtype of the TensorTrainSlice
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
            if raw.is_canonical(self.tomatrices_unchecked(),center):
                self._center=center
                return True
            else:
                return False
        else:
            self._center=raw.find_orthogonality_center(self.tomatrices_unchecked())
            return self._center is not None

    def canonicalize(self,center=None,copy=False,qr=la.qr):
        '''
            Canonicalizes the TensorTrainSlice
        '''
        if copy:
            ret=self.copy()
        else:
            ret=self
        if center is None:
            center=self.L-1
        elif center<0:
            center=self.L+center
        if self._center is None:
            raw.canonicalize(ret.tomatrices_unchecked(),center,qr=qr)
            ret._center=center
        else:
            raw.shift_orthogonality_center(ret.tomatrices_unchecked(),self._center,center,qr=qr)
            ret._center=center
        return ret
    def singular_values(self,left=0,right=-1,ro=False,qr=la.qr,svd=la.svd):
        if ro:
            ret=self.copy()
        else:
            ret=self
        if right<0:
            right=self.L+right
        if ret._center is None:
            ret.canonicalize(right,qr=qr)
        return raw.singular_values(ret.tomatrices_unchecked()[left:right+1],ret.center-left,svd=svd)

    @classmethod
    def frommatrices(cls,matrices):
        '''
            Build a TensorTrainSlice from a list of matrices with the internal
            indices being first and last. Copys the matrices to prevent violation
            of invariants through references outside of the new instance.
            See also:
            =========
            frommatrices_slice
            TensorTrainSlice.frommatrices_unchecked
            TensorTrainArray.frommatrices
        '''
        return cls([x.copy() for x in matrices])

    @classmethod
    def frommatrices_unchecked(cls,matrices):
        '''
            Build a TensorTrainSlice from a list of matrices with the internal
            indices being first and last. Doesn't copy matrices, reuse of old
            references can lead to violation of invariants
            See also:
            =========
            frommatrices_slice_unchecked
            TensorTrainSlice.frommatrices
            TensorTrainArray.frommatrices_unchecked

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
    def tomatrices(self):
        '''
            Returns a copy of the list of underlying matrices.
        '''
        return [x.copy() for x in self._data]
    def tomatrices_unchecked(self):
        '''
            Returns the list of matrices which make up the TensorTrainSlice.
            Doesn't copy, if invariants are violated that is your problem.
        '''
        return self._data
    def setmatrices(self,dat):
        bup=self._data
        self._data=[x.copy() for x in self.dat]
        try:
            self._check_consistency()
            self.clearcenter()
        except ValueError as e:
            self._data=bup #restore old values in case of error
            raise e

    def setmatrices_unchecked(self,dat):
        '''
            Sets the data for this instance of TensorTrainSlice, doesn't perform any checks, nor copying
        '''
        self._data=dat

    def __array_function__(self,func,types,args,kwargs):
        f=HANDLER_FUNCTION_SLICE.get(func.__name__,None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def __array_ufunc__(self,ufunc,method,*args,**kwargs):
        '''
            Implements the numpy __array_ufunc__ protocol for TensorTrainSlice
        '''
        f=HANDLER_UFUNC_SLICE.get((ufunc.__name__,method),None)
        if f is None:
            return NotImplemented
        return f(*args,**kwargs)
    def transpose(self,*axes):
        r=len(self.shape)
        axes=_normalize_axes(axes)
        if axes[0]!=0 and axes[-1]!=r:
            raise NotImplementedError("transposing of the boundaries of a TensorTrainSlice is not supported yet")
        else:
            return self.__class__.frommatrices_unchecked([x.transpose(axes) for x in a.M])
    def recluster(self,newcluster=None,axes=None,copy=False):
        if newcluster is None:
            newcluster = raw.find_balanced_cluster(self.shape)
        if axes is not None:
            if len(axes)==0:
                pass
            elif max(axes)>=len(self.shape)-1:
                raise ValueError("Axis %i out of bounds for TensorTrainSlice with %i dimensions"%(max(axes),len(self.shape)))
            elif _check_cluster_axes(self.cluster,newcluster,axes):
                if copy:
                    return self.copy()
                return self
            oaxes=[i for i in range(1,len(self.shape)-1) if i not in axes]
            ocluster=raw.find_balanced_cluster(tuple(self.shape[s] for s in oaxes),len(newcluster))
            ncluster=[]
            for nc,oc in zip(newcluster,ocluster):
                ncc=[0]*(len(self.shape)-2)
                for i,c in zip(oaxes,oc):
                    ncc[i-1]=c
                for i,c in zip(axes,nc):
                    ncc[i-1]=c
                ncluster.append(tuple(ncc))
            newcluster=ncluster
        else:
            if newcluster==self.cluster:
                if copy:
                    return self.copy()
                return self

        rank=len(self.shape)-2
        for n in newcluster:
            if len(n)!=rank:
                raise ValueError("cluster does not have rank %i"%rank)
        tsh=tuple([_product(c[d] for c in newcluster) for d in range(rank)])
        if tsh!=self.shape[1:-1]:
            raise ValueError("cluster %s not compatible with shape %s"%(newcluster,self.shape))


        if not copy:
            self.clearcenter()
            self.setmatrices_unchecked(raw.recluster(self.tomatrices_unchecked(),newcluster))
            return self
        else:
            return self.__class__.frommatrices(raw.recluster(self.tomatrices(),newcluster))
    def truncate(self,chi_max=None,cutoff=0.0,left=0,right=-1,qr=la.qr,svd=la.svd):
        if right<0:
            right+=self.L
        self.canonicalize(right)
        mats=self.tomatrices_unchecked()[left:right+1]
        raw.left_truncate_svd(mats,chi_max,cutoff)
        self.tomatrices_unchecked()[left:right+1]=mats
        assert self.is_canonical(left)
        self._center=left
    def copy(self):
        return self.__class__.frommatrices([x.copy() for x in self.tomatrices_unchecked()])
    def conj(self):
        return self.__class__.frommatrices([x.conj() for x in self.tomatrices_unchecked()])
