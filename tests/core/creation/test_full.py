import numpy as np
from ... import check_ttarray_dense,check_ttslice_dense
from ttarray import full,full_like,full_slice
import pytest
ARRAY_PROTOTYPE=full((1,),5,int,((1,),),2)
SLICE_PROTOTYPE=full_slice((2,2,3),3.1459,float,((2,),),2)
DTYPE_VAL=[(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5),(complex,3)]
def test_full_ttarray(shape_cluster):
    shape,cluster=shape_cluster
    for dt,val in DTYPE_VAL:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full(shape,val,dt,cluster,chi)
            ar2=full_like(ARRAY_PROTOTYPE,val,dt,shape=shape,cluster=cluster,chi=chi)
            ar3=full_like(ar,val)
            ar4=np.full_like(ar,val)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            npar=np.full(shape,val,dtype=dt)
            check_ttarray_dense(ar,npar,cluster,chi)
            check_ttarray_dense(ar2,npar,cluster,chi)
            check_ttarray_dense(ar3,npar,cluster,chi)
            check_ttarray_dense(ar4,npar,cluster,chi)
def test_full_ttslice(shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    for dt,val in DTYPE_VAL:
        for chi in [1,2,tuple(range(1,len(cluster)))]:
            ar=full_slice(shape,val,dt,cluster,chi)
            ar2=full_like(SLICE_PROTOTYPE,val,dt,shape=shape,cluster=cluster,chi=chi)
            ar3=full_like(ar,val)
            ar4=np.full_like(ar,val)
            if isinstance(chi,int):
                chi=tuple([chi]*(len(cluster)-1))
            npar=np.full(shape,val,dtype=dt)
            check_ttslice_dense(ar,npar,cluster,chi)
            check_ttslice_dense(ar2,npar,cluster,chi)
            check_ttslice_dense(ar3,npar,cluster,chi)
            check_ttslice_dense(ar4,npar,cluster,chi)

def test_full_ttarray_nocluster(shape):
    shape,cluster=shape
    chi=tuple([1]*(len(cluster)-1))
    for dt,val in DTYPE_VAL:
        ar=full(shape,val,dt)
        ar2=np.full(shape,val,dt,like=ar)
        ar3=full_like(ARRAY_PROTOTYPE,val,dt,shape)
        ar4=np.full_like(ARRAY_PROTOTYPE,val,dt,shape=shape)
        ar5=np.full_like(ar,val)
        npar=np.full(shape,val,dtype=dt)
        check_ttarray_dense(ar,npar,cluster,chi)
        check_ttarray_dense(ar2,npar,cluster,chi)
        check_ttarray_dense(ar3,npar,cluster,chi)
        check_ttarray_dense(ar4,npar,cluster,chi)
        check_ttarray_dense(ar5,npar,cluster,chi)

def test_full_ttslice_nocluster(shape):
    shape,cluster=shape
    shape=tuple([2]+list(shape)+[3])
    chi=tuple([1]*(len(cluster)-1))
    for dt,val in DTYPE_VAL:
        ar=full_slice(shape,val,dt)
        ar2=np.full(shape,val,dt,like=ar)
        ar3=full_like(SLICE_PROTOTYPE,val,dt,shape)
        ar4=np.full_like(SLICE_PROTOTYPE,val,dt,shape=shape)
        ar5=np.full_like(ar,val)
        npar=np.full(shape,val,dtype=dt)
        check_ttslice_dense(ar,npar,cluster,chi)
        check_ttslice_dense(ar2,npar,cluster,chi)
        check_ttslice_dense(ar3,npar,cluster,chi)
        check_ttslice_dense(ar4,npar,cluster,chi)
        check_ttslice_dense(ar5,npar,cluster,chi)
