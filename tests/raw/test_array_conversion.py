from ttarray.raw import array_to_ttslice,ttslice_to_array
from ttarray.raw import is_canonical,find_balanced_cluster,trivial_decomposition
import numpy.linalg as la
import numpy as np
import pytest
def test_array_conversion_trivial(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    ar=np.random.random(size=shape)
    ttar=array_to_ttslice(ar,cluster,trivial_decomposition)
    assert ttslice_to_array(ttar)==pytest.approx(ar)
def test_array_conversion_qr(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    ar=np.random.random(size=shape)
    ttar=array_to_ttslice(ar,cluster,la.qr)
    assert is_canonical(ttar)
    assert ttslice_to_array(ttar)==pytest.approx(ar)
def test_array_conversion_rq(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    ar=np.random.random(size=shape)
    def rq(x):
        q,r=la.qr(x.T)
        return r.T,q.T
    ttar=array_to_ttslice(ar,cluster,rq)
    assert is_canonical(ttar,center=0)
    assert ttslice_to_array(ttar)==pytest.approx(ar)
def test_balanced_cluster(shape):
    assert find_balanced_cluster(shape[0])==shape[1]
