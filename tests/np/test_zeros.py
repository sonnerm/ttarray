import numpy as np
from ttarray.np import zeros,zeros_like,zeros_slice
def test_zeros_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=zeros(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert (arb==np.zeros(shape,dtype=dt)).all()
def test_zeros_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=zeros_slice(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert (arb==np.zeros(shape,dtype=dt)).all()

def test_zeros_ttarray_nocluster(shape):
    shape,cluster=shape
    for dt in [complex,float,int]:
        ar=zeros(shape,dt)
        arnp=np.zeros(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==tuple([1]*(len(cluster)-1))
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.zeros(shape,dtype=dt)).all()
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.zeros(shape,dtype=dt)).all()

def test_zeros_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        ar=zeros_slice(shape,dt)
        # arnp=np.zeros(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==tuple([1]*(len(cluster)-1))
        # assert arnp.shape==shape
        # assert arnp.cluster==cluster
        # assert arnp.chi==tuple([1]*(len(cluster)-1))
        # arb=np.array(arnp)
        # assert type(arb)==np.ndarray
        # assert arb.shape==shape
        # assert (arb==np.zeros(shape,dtype=dt)).all()
        #doesn't work since like keyword arg is not passed through numpy :(
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.zeros(shape,dtype=dt)).all()
