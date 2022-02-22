import numpy as np
from collection import defaultdict
HANDLER_FUNCTION["dense"]=defaultdict(dense_array_function)
HANDLER_UFUNC["dense"]=defaultdict(dense_array_ufunc)
def dense_array_function(func,types,args,kwargs):
    args=[np.asarray(a) if (isinstance(a,MatrixProductSlice) or isinstance(a,MatrixProductArray)) for a in args]
    return func(*args,**kwargs)
def dense_array_ufunc(ufunc,method,args,kwargs):
    args=[np.asarray(a) if (isinstance(a,MatrixProductSlice) or isinstance(a,MatrixProductArray)) for a in args]
    return getattr(ufunc,method)(*args,**kwargs)
