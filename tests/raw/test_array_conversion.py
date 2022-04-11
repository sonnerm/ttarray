import ttarray.raw as raw
from .. import DENSE_SHAPE,TINY_SHAPE
import numpy.linalg as la
import numpy as np
import pytest
def test_array_conversion_trivial(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=tuple([1]+list(shape)+[1])
        for cluster in cls:
            ar=np.random.random(size=shape)
            ttar=raw.dense_to_ttslice(ar,cluster,raw.trivial_decomposition)
            assert raw.ttslice_to_dense(ttar)==pytest.approx(ar)
            assert len(ttar)==len(cluster)
            assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
            assert ttar[0].shape[0] == 1
            assert ttar[-1].shape[-1] == 1

def test_array_conversion_qr(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        for cluster in cls:
            ar=np.random.random(size=shape)
            ttar=raw.dense_to_ttslice(ar,cluster,la.qr)
            assert raw.is_canonical(ttar,-1)
            assert raw.is_left_canonical(ttar)
            assert raw.ttslice_to_dense(ttar)==pytest.approx(ar)
            assert len(ttar)==len(cluster)
            assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
            assert ttar[0].shape[0] == 2
            assert ttar[-1].shape[-1] == 3
def test_array_conversion_rq(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(2,)
        for cluster in cls:
            ar=np.random.random(size=shape)
            def rq(x):
                q,r=la.qr(x.T)
                return r.T,q.T
            ttar=raw.dense_to_ttslice(ar,cluster,rq)
            assert raw.is_canonical(ttar,0)
            assert raw.is_right_canonical(ttar)
            assert raw.ttslice_to_dense(ttar)==pytest.approx(ar)
            assert len(ttar)==len(cluster)
            assert [ta.shape[1:-1]==c for ta,c in zip(ttar,cluster)]
            assert ttar[0].shape[0] == 2
            assert ttar[-1].shape[-1] == 2
def test_balanced_cluster():
    for shape,cls in DENSE_SHAPE.items():
        assert raw.find_balanced_cluster(shape)==cls[0]
