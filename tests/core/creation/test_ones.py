import numpy as np
from ttarray import ones,ones_like,ones_slice
from ... import check_ttarray_dense,check_ttslice_dense
ARRAY_PROTOTYPE=ones((1,),int,((1,),),2)
SLICE_PROTOTYPE=ones_slice((2,2,3),int,((2,),),2)
def test_ones_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=ones(shape,dt,cluster,chi)
            ar2=ones_like(ARRAY_PROTOTYPE,dt,shape=shape,cluster=cluster,chi=chi)
            ar3=ones_like(ar)
            ar4=np.ones_like(ar)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            npar=np.ones(shape,dtype=dt)
            check_ttarray_dense(ar,npar,cluster,chi)
            check_ttarray_dense(ar2,npar,cluster,chi)
            check_ttarray_dense(ar3,npar,cluster,chi)
            check_ttarray_dense(ar4,npar,cluster,chi)
def test_ones_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt in [complex,float,int]:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=ones_slice(shape,dt,cluster,chi)
            ar2=ones_like(SLICE_PROTOTYPE,dt,shape=shape,cluster=cluster,chi=chi)
            ar3=ones_like(ar)
            ar4=np.ones_like(ar)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            npar=np.ones(shape,dtype=dt)
            check_ttslice_dense(ar,npar,cluster,chi)
            check_ttslice_dense(ar2,npar,cluster,chi)
            check_ttslice_dense(ar3,npar,cluster,chi)
            check_ttslice_dense(ar4,npar,cluster,chi)

def test_ones_ttarray_nocluster(shape):
    shape,cluster=shape
    chi=tuple([1]*(len(cluster)-1))
    for dt in [complex,float,int]:
        ar=ones(shape,dt)
        ar2=np.ones(shape,dt,like=ar)
        ar3=ones_like(ARRAY_PROTOTYPE,dt,shape)
        ar4=np.ones_like(ARRAY_PROTOTYPE,dt,shape=shape)
        ar5=np.ones_like(ar)
        npar=np.ones(shape,dtype=dt)
        check_ttarray_dense(ar,npar,cluster,chi)
        check_ttarray_dense(ar2,npar,cluster,chi)
        check_ttarray_dense(ar3,npar,cluster,chi)
        check_ttarray_dense(ar4,npar,cluster,chi)
        check_ttarray_dense(ar5,npar,cluster,chi)

def test_ones_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    chi=tuple([1]*(len(cluster)-1))
    for dt in [complex,float,int]:
        ar=ones_slice(shape,dt)
        ar2=np.ones(shape,dt,like=ar)
        ar3=ones_like(SLICE_PROTOTYPE,dt,shape)
        ar4=np.ones_like(SLICE_PROTOTYPE,dt,shape=shape)
        ar5=np.ones_like(ar)
        npar=np.ones(shape,dtype=dt)
        check_ttslice_dense(ar,npar,cluster,chi)
        check_ttslice_dense(ar2,npar,cluster,chi)
        check_ttslice_dense(ar3,npar,cluster,chi)
        check_ttslice_dense(ar4,npar,cluster,chi)
        check_ttslice_dense(ar5,npar,cluster,chi)
