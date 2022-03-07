import numpy as np
from ttarray import empty,empty_like,empty_slice
def test_empty_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=empty(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
def test_empty_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=empty_slice(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape

def test_empty_ttarray_nocluster(shape):
    shape,cluster=shape
    for dt in [complex,float,int]:
        ar=empty(shape,dt)
        arnp=np.empty(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape

def test_empty_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        ar=empty_slice(shape,dt)
        arnp=np.empty(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
