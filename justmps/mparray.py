from numpy.lib.mixins import NDArrayOperatorsMixin
from .mpslice import MatrixProductSlice
from .dispatch import _dispatch_array_ufunc
from .dispatch import _dispatch_array_function
class MatrixProductArray(NDArrayOperatorsMixin):
    def __init__(self,mpas):
        if mpas.shape[0]!=1 or mpas.shape[-1]!=1:
            raise ValueError("MatrixProductArrays cannot have a non-contracted boundary")
        self._mpas=mpas
        self.shape=mpas.shape[1:-1]
        self.dtype=mpas.dtype
    def __array__(self,dtype=None):
        return np.asarray(self._mpas,dtype)[0,...,0]
    @classmethod
    def from_array(cls,ar,dims=None,canonicalize=True):
        return cls(MatrixProductSlice.from_array(ar,dims,canonicalize))
    @classmethod
    def from_matrix_list(cls,mpl):
        return cls(MatrixProductSlice(mpl))
    @classmethod
    def from_matrix_product_slice(self,mps):
        return cls(mps)
    def as_matrix_product_slice(self):
        return copy.copy(self._mpas) #shallow copy necessary to protect invariants
    def as_matrix_list(self):
        return self._mpas.as_matrix_list() #already does shallow copying
    def __array_function__(self,func,types,args,kwargs):
        return _dispatch_array_function(func,types,args,kwargs)
    def __array_ufunc__(self,ufunc,method,args,kwargs):
        return _dispatch_array_ufunc(ufunc,method,args,kwargs)
    def truncate(self,**kwargs):
        self._mpas=self._mpas.truncate(**kwargs)
