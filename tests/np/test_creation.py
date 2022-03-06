import numpy as np
from ttarray.np import zeros,zeros_like
def test_zeros_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt in [complex,float,int]:
        for chi in [1,2,list(range(1,len(cluster)))]:
            ar=zeros(shape,dt,cluster,chi)
            if isinstance(chi,int):
                chi=[chi]*(len(cluster)-1)
            assert ar.shape==shape
            assert ar.cluster==cluster
            assert ar.chi==chi
            arb=np.array(ar) #back conversion
            assert type(arb)==np.ndarray
            assert arb.shape==shape
            assert (arb==np.zeros(shape,dtype=dt)).all()

# def test_zeros_ttslice(shape_cluster_slice):
#     shape,cluster=shape_cluster_slice
#     for dt in [complex,float,int]:
#         for chi in [1,2,list(range(1,len(cluster)))]:
#             ar=zeros(shape,dt,cluster,chi)
#             assert ar.shape==shape
#             assert ar.cluster==cluster
#             assert ar.chi==chi
#             arb=np.array(ar) #back conversion
#             assert type(arb)==np.ndarray
#             assert arb.shape==shape
#             assert (arb==np.zeros(shape,dtype=dt)).all()
def test_zeros_ttarray_nocluster(shape):
    shape,cluster=shape
    for dt in [complex,float,int]:
        ar=zeros(shape,dt)
        arnp=np.zeros(shape,dt,like=ar)
        assert ar.shape==shape
        assert ar.cluster==cluster
        assert ar.chi==[1]*(len(cluster)-1)
        assert arnp.shape==shape
        assert arnp.cluster==cluster
        assert arnp.chi==[1]*(len(cluster)-1)
        arb=np.array(arnp)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.zeros(shape,dtype=dt)).all()
        arb=np.array(ar)
        assert type(arb)==np.ndarray
        assert arb.shape==shape
        assert (arb==np.zeros(shape,dtype=dt)).all()

# def test_zeros_ttslice_nocluster(shape_slice):
#     shape,cluster=shape
#     for dt in [complex,float,int]:
#         ar=zeros(shape,dt,cluster)
#         arnp=np.zeros(shape,like=like)
#         assert ar.shape==shape
#         assert ar.cluster=cluster
#         arb=np.array(ar)
#         assert type(arb)==np.ndarray
#         assert arb.shape==shape
#         assert (arb==np.zeros(shape,dtype=dt)).all()
