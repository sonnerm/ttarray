import numpy as np
from ttarray.np import full,full_like,full_slice
import pytest
pytestmark=pytest.mark.skip
def test_full_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert arb==pytest.approx(np.full(shape,dtype=dt))
def test_full_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full_slice(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert arb==pytest.approx(np.full(shape,dtype=dt))

def test_full_ttarray_nocluster(shape):
    shape,cluster=shape
    for dt in [complex,float,int]:
        ar=full(shape,dt)
        arnp=np.full(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.full(shape,dtype=dt)).all()
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.full(shape,dtype=dt)).all()
def test_full_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        ar=full_slice(shape,dt)
        arnp=np.full(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.full(shape,dtype=dt)).all()
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert arb==pytest.approx(np.full(shape,dtype=dt)) #there is a division ...
