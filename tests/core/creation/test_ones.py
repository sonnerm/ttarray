import numpy as np
import ttarray as tt
from ... import check_dense,check_constant
from ... import DENSE_SHAPE,TINY_SHAPE,LARGE_SHAPE
import pytest
def test_ones_ttarray():
    for shape,cls in DENSE_SHAPE.items():
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.ones(shape)
        ar2=np.ones(shape,like=ar)
        ar3=np.ones_like(ar)
        npar=np.ones(shape)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.ones(shape,cluster=cluster,chi=chi)
                ar2=tt.ones_like(ar)
                ar3=np.ones_like(ar)
                chi=tuple([chi]*(len(cluster)-1))
                npar=np.ones(shape)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
                check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
def test_ones_ttslice():
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.ones_slice(shape)
        ar2=np.ones(shape,like=ar)
        ar3=np.ones_like(ar)
        npar=np.ones(shape)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.ones_slice(shape,float,cluster,chi)
                ar2=tt.ones_like(ar)
                ar3=np.ones_like(ar)
                chi=tuple([chi]*(len(cluster)-1))
                npar=np.ones(shape)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
                check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)

def test_ones_ttarray_type():
    for dt in [int,complex,bool]: #float is already tested
        for shape,cls in TINY_SHAPE.items():
            ar=tt.ones(shape,dt)
            ar2=np.ones(shape,dtype=dt,like=ar)
            ar3=np.ones_like(ar)
            npar=np.ones(shape,dtype=dt)
            chi=tuple([1]*(len(cls[0])-1))
            check_dense(ar,npar,cls[0],chi,tt.TensorTrainArray)
            check_dense(ar2,npar,cls[0],chi,tt.TensorTrainArray)
            check_dense(ar3,npar,cls[0],chi,tt.TensorTrainArray)
            for cluster in cls:
                chi=tuple([1]*(len(cluster)-1))
                ar=tt.ones(shape,dt,cluster)
                ar2=tt.ones_like(ar)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)

def test_ones_ttslice_type():
    for dt in [int,complex,bool]: #float is already tested
        for shape,cls in TINY_SHAPE.items():
            shape=(2,)+shape+(3,)
            ar=tt.ones_slice(shape,dt)
            ar2=np.ones(shape,dtype=dt,like=ar)
            ar3=np.ones_like(ar)
            npar=np.ones(shape,dtype=dt)
            chi=tuple([1]*(len(cls[0])-1))
            check_dense(ar,npar,cls[0],chi,tt.TensorTrainSlice)
            check_dense(ar2,npar,cls[0],chi,tt.TensorTrainSlice)
            check_dense(ar3,npar,cls[0],chi,tt.TensorTrainSlice)
            for cluster in cls:
                chi=tuple([1]*(len(cluster)-1))
                ar=tt.ones_slice(shape,dt,cluster)
                ar2=tt.ones_like(ar)
                check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
                check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
def test_ones_ttarray_large():
    val=1.0
    for shape,cls in LARGE_SHAPE.items():
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.ones(shape,)
        ar2=np.ones(shape,like=ar)
        ar3=np.ones_like(ar)
        check_constant(ar,val,shape,cluster,chi,tt.TensorTrainArray)
        check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainArray)
        check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.ones(shape,cluster=cluster,chi=chi)
                ar2=tt.ones_like(ar)
                ar3=np.ones_like(ar)
                chi=tuple([chi]*(len(cluster)-1))
                check_constant(ar,val,shape,cluster,chi,tt.TensorTrainArray)
                check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainArray)
                check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainArray)

def test_ones_ttslice_large():
    val=1.0
    for shape,cls in LARGE_SHAPE.items():
        shape=(2,)+shape+(3,)
        cluster=cls[0]
        chi=tuple([1]*(len(cluster)-1))
        ar=tt.ones_slice(shape,)
        ar2=np.ones(shape,like=ar)
        ar3=np.ones_like(ar)
        check_constant(ar,val,shape,cluster,chi,tt.TensorTrainSlice)
        check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainSlice)
        check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainSlice)
        for cluster in cls:
            for chi in [1,2]:
                ar=tt.ones_slice(shape,cluster=cluster,chi=chi)
                ar2=tt.ones_like(ar)
                ar3=np.ones_like(ar)
                chi=tuple([chi]*(len(cluster)-1))
                check_constant(ar,val,shape,cluster,chi,tt.TensorTrainSlice)
                check_constant(ar2,val,shape,cluster,chi,tt.TensorTrainSlice)
                check_constant(ar3,val,shape,cluster,chi,tt.TensorTrainSlice)
