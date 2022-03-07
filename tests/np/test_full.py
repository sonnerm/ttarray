import numpy as np
from ttarray.np import full,full_like,full_slice
import pytest
def test_full_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt,val in [(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5)]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full(shape,val,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert arb==pytest.approx(np.full(shape,val,dtype=dt))
def test_full_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt,val in [(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5)]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full_slice(shape,val,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert arb==pytest.approx(np.full(shape,val,dtype=dt))

def test_full_ttarray_nocluster(shape):
    shape,cluster=shape
    for dt,val in [(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5)]:
        ar=full(shape,val,dt)
        arnp=np.full(shape,val,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert arb==pytest.approx(np.full(shape,val,dtype=dt))
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert arb==pytest.approx(np.full(shape,val,dtype=dt))
def test_full_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    for dt,val in [(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5)]:
        ar=full_slice(shape,val,dt)
        arnp=np.full(shape,val,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert arb==pytest.approx(np.full(shape,val,dtype=dt))
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert arb==pytest.approx(val,np.full(shape,val,dtype=dt)) #there is a division ...
