import numpy as np
import ttarray as tt
from ... import check_dense,check_constant
from ... import DENSE_SHAPE,TINY_SHAPE,LARGE_SHAPE
import pytest
DTYPE_VAL=[(complex,np.pi+np.e*1.0j),(float,3.1459),(int,5),(complex,3)]
def test_full_ttarray():
    val=np.e+np.pi*1.0j
    for shape,cls in DENSE_SHAPE.items():
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.full(shape,val)
        ar2=np.full(shape,val,like=ar)
        ar3=np.full_like(ar,val)
        npar=np.full(shape,val)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.full(shape,val,cluster=cluster,chi=chi)
                ar2=tt.full_like(ar,val)
                ar3=np.full_like(ar,val)
                chi=tuple([chi]*(len(cluster)-1))
                npar=np.full(shape,val)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
                check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
def test_full_ttslice():
    val=np.e
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.full_slice(shape,val)
        ar2=np.full(shape,val,like=ar)
        ar3=np.full_like(ar,val)
        npar=np.full(shape,val)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.full_slice(shape,val,float,cluster,chi)
                ar2=tt.full_like(ar,val)
                ar3=np.full_like(ar,val)
                chi=tuple([chi]*(len(cluster)-1))
                npar=np.full(shape,val)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
                check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
def test_full_ttslice_type():
    for shape,cls in TINY_SHAPE.items():
        shape=(2,)+shape+(3,)
        for dt,val in DTYPE_VAL:
            cluster=cls[0]
            chi=tuple([1]*(len(cluster)-1))
            ar=tt.full_slice(shape,val,dt)
            ar2=np.full(shape,val,dt,like=ar)
            ar3=np.full_like(ar,val)
            npar=np.full(shape,val,dt)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
            check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
            check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
            for cluster in cls:
                for chi in [1,2]:
                    ar=tt.full_slice(shape,val,dt,cluster=cluster,chi=chi)
                    ar2=tt.full_like(ar,val)
                    ar3=np.full_like(ar,val)
                    chi=tuple([chi]*(len(cluster)-1))
                    npar=np.full(shape,val,dt)
                    check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
                    check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
                    check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
def test_full_ttarray_type():
    for shape,cls in TINY_SHAPE.items():
        for dt,val in DTYPE_VAL:
            cluster=cls[0]
            chi=tuple([1]*(len(cluster)-1))
            ar=tt.full(shape,val,dt)
            ar2=np.full(shape,val,dt,like=ar)
            ar3=np.full_like(ar,val)
            npar=np.full(shape,val,dt)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
            check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
            check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
            for cluster in cls:
                for chi in [1,2]:
                    ar=tt.full(shape,val,dt,cluster=cluster,chi=chi)
                    ar2=tt.full_like(ar,val)
                    ar3=np.full_like(ar,val)
                    chi=tuple([chi]*(len(cluster)-1))
                    npar=np.full(shape,val,dt)
                    check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
                    check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
                    check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
def test_full_ttarray_large():
    val=4
    for shape,cls in LARGE_SHAPE.items():
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.full(shape,val)
        ar2=np.full(shape,val,like=ar)
        ar3=np.full_like(ar,val)
        check_constant(ar,val,shape,cluster,chi,tt.TensorTrainArray)
        check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainArray)
        check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.full(shape,val,cluster=cluster,chi=chi)
                ar2=tt.full_like(ar,val)
                ar3=np.full_like(ar,val)
                chi=tuple([chi]*(len(cluster)-1))
                check_constant(ar,val,shape,cluster,chi,tt.TensorTrainArray)
                check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainArray)
                check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainArray)

def test_full_ttslice_large():
    val=True
    for shape,cls in LARGE_SHAPE.items():
        shape=(2,)+shape+(3,)
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.full_slice(shape,val)
        ar2=np.full(shape,val,like=ar)
        ar3=np.full_like(ar,val)
        check_constant(ar,val,shape,cluster,chi,tt.TensorTrainSlice)
        check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainSlice)
        check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainSlice)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.full_slice(shape,val,cluster=cluster,chi=chi)
                ar2=tt.full_like(ar,val)
                ar3=np.full_like(ar,val)
                chi=tuple([chi]*(len(cluster)-1))
                check_constant(ar,val,shape,cluster,chi,tt.TensorTrainSlice)
                check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainSlice)
                check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainSlice)
