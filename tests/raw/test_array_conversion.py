from ttarray.raw import dense_to_ttslice,ttslice_to_dense
from ttarray.raw import is_canonical,find_balanced_cluster,trivial_decomposition
from ttarray.raw import is_left_canonical,is_right_canonical
import numpy.linalg as la
import numpy as np
import pytest
def test_array_conversion_trivial(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([1]+list(shape)+[1])
    ar=np.random.random(size=shape)
    ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
    assert ttslice_to_dense(ttar)==pytest.approx(ar)
    assert len(ttar)==len(cluster)
    assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
    assert ttar[0].shape[0] == 1
    assert ttar[-1].shape[-1] == 1

def test_array_conversion_qr(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    ar=np.random.random(size=shape)
    ttar=dense_to_ttslice(ar,cluster,la.qr)
    assert is_canonical(ttar,-1)
    assert is_left_canonical(ttar)
    assert ttslice_to_dense(ttar)==pytest.approx(ar)
    assert len(ttar)==len(cluster)
    assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
    assert ttar[0].shape[0] == 2
    assert ttar[-1].shape[-1] == 3
def test_array_conversion_rq(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    ar=np.random.random(size=shape)
    def rq(x):
        q,r=la.qr(x.T)
        return r.T,q.T
    ttar=dense_to_ttslice(ar,cluster,rq)
    assert is_canonical(ttar,0)
    assert is_right_canonical(ttar)
    assert ttslice_to_dense(ttar)==pytest.approx(ar)
    assert len(ttar)==len(cluster)
    assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
    assert ttar[0].shape[0] == 2
    assert ttar[-1].shape[-1] == 3
def test_balanced_cluster(shape):
    assert find_balanced_cluster(shape[0])==shape[1]
