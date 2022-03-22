import numpy as np
import pytest
from ttarray import TensorTrainArray,TensorTrainSlice
import functools
def check_ttarray_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    assert isinstance(tt,TensorTrainArray)
    assert tt.cluster == cluster
    if chi is not None:
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
    if chi is not None:
        assert tt.chi == chi
    assert tt.shape == ar.shape
    assert tt.dtype == ar.dtype
    arb=np.array(tt)
    assert type(arb)==np.ndarray
    assert arb == pytest.approx(ar)
def check_raw_ttslice_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    check_ttslice_dense(TensorTrainSlice.frommatrices(tt),ar,cluster,chi)
def random_array(shape,dtype):
    if np.dtype(dtype).kind=="i":
        return np.random.randint(-10000000,10000000,size=shape,dtype=dtype)
    elif np.dtype(dtype).kind=="f":
        return np.array(np.random.randn(*shape)-1,dtype=dtype)
    elif np.dtype(dtype).kind=="c":
        return np.array(np.random.randn(*shape)+1.0j*np.random.randn(*shape),dtype=dtype)
    else:
        raise NotImplementedError("unknown dtype")

def _product(seq):
    return functools.reduce(lambda x,y:x*y, seq,1)
def calc_chi(cluster,lefti=1,righti=1):
    left,right=[lefti],[righti]
    for c in cluster:
        left.append(left[-1]*_product(c))
    for c in cluster[::-1]:
        right.append(right[-1]*_product(c))
    return tuple(min(l,r) for l,r in zip(left[1:-1],right[1:-1][::-1]))
