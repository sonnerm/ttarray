import numpy
import pytest
from ttarray import TensorTrainArray,TensorTrainSlice
def check_ttarray_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    assert isinstance(tt,TensorTrainArray)
    assert tt.cluster == cluster
    assert tt.chi == chi
    assert tt.shape == ar.shape
    assert tt.dtype == ar.dtype
    arb=numpy.array(tt)
    assert type(arb)==numpy.ndarray
    assert arb == pytest.approx(ar)

def check_ttslice_dense(tt,ar,cluster,chi):
    __traceback_hide__ = True
    assert isinstance(tt,TensorTrainSlice)
    assert tt.cluster == cluster
    assert tt.chi == chi
    assert tt.shape == ar.shape
    assert tt.dtype == ar.dtype
    arb=numpy.array(tt)
    assert type(arb)==numpy.ndarray
    assert arb == pytest.approx(ar)
