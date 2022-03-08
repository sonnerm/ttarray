import numpy as np
import pytest
from ttarray import TensorTrainArray,TensorTrainSlice
def check_ttarray_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    assert isinstance(tt,TensorTrainArray)
    assert tt.cluster == cluster
    assert tt.chi == chi
    assert tt.shape == ar.shape
    assert tt.dtype == ar.dtype
    arb=np.array(tt)
    print(arb)
    assert type(arb)==np.ndarray
    assert arb == pytest.approx(ar)

def check_ttslice_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    assert isinstance(tt,TensorTrainSlice)
    assert tt.cluster == cluster
    assert tt.chi == chi
    assert tt.shape == ar.shape
    assert tt.dtype == ar.dtype
    arb=np.array(tt)
    assert type(arb)==np.ndarray
    assert arb == pytest.approx(ar)
def random_array(shape,dtype):
    if np.dtype(dtype).kind=="i":
        return np.random.randint(-10000000,10000000,size=shape,dtype=dtype)
    elif np.dtype(dtype).kind=="f":
        return np.array(np.random.randn(*shape)-1,dtype=dtype)
    elif np.dtype(dtype).kind=="c":
        return np.array(np.random.randn(*shape)+1.0j*np.random.randn(*shape),dtype=dtype)
    else:
        raise NotImplementedError("unknown dtype")
