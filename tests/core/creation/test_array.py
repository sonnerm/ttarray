import numpy as np
import ttarray as tt
from ... import check_dense,random_array
from ... import DENSE_SHAPE
import pytest
import functools
def _product(seq):
    return functools.reduce(lambda x,y:x*y, seq,1)
def _calc_chi(cluster,lefti=1,righti=1):
    left,right=[lefti],[righti]
    for c in cluster:
        left.append(left[-1]*_product(c))
    for c in cluster[::-1]:
        right.append(right[-1]*_product(c))
    return tuple(min(l,r) for l,r in zip(left[1:-1],right[1:-1][::-1]))

def test_ttarray_ndarray(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        cluster=cls[0]
        chi=_calc_chi(cluster)
        npar=random_array(shape,int)
        ar=tt.array(npar)
        ar2=np.array(npar,like=ar)
        ar3=np.asarray(npar,like=ar)
        ar4=np.asanyarray(npar,like=ar)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar4,npar,cluster,chi,tt.TensorTrainArray)
        ar=tt.array(npar,dtype=complex)
        ar2=np.array(npar,dtype=complex,like=ar)
        ar3=np.asarray(npar,dtype=complex,like=ar)
        ar4=np.asanyarray(npar,dtype=complex,like=ar)
        npar2=np.array(npar,dtype=complex)
        check_dense(ar,npar2,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar2,cluster,chi,tt.TensorTrainArray)
        check_dense(ar3,npar2,cluster,chi,tt.TensorTrainArray)
        check_dense(ar4,npar2,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            chi=_calc_chi(cluster)
            npar=random_array(shape,float)
            ar=tt.array(npar,cluster=cluster)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)

def test_ttslice_ndarray(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=(3,)+shape+(2,)
        cluster=cls[0]
        chi=_calc_chi(cluster,3,2)
        npar=random_array(shape,float)
        ar=tt.slice(npar)
        ar2=np.array(npar,like=ar)
        ar3=np.asarray(npar,like=ar)
        ar4=np.asanyarray(npar,like=ar)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar3,npar,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar4,npar,cluster,chi,tt.TensorTrainSlice)
        ar=tt.slice(npar,dtype=complex)
        ar2=np.array(npar,dtype=complex,like=ar)
        ar3=np.asarray(npar,dtype=complex,like=ar)
        ar4=np.asanyarray(npar,dtype=complex,like=ar)
        npar2=np.array(npar,dtype=complex)
        check_dense(ar,npar2,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar2,npar2,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar3,npar2,cluster,chi,tt.TensorTrainSlice)
        check_dense(ar4,npar2,cluster,chi,tt.TensorTrainSlice)
        for cluster in cls:
            chi=_calc_chi(cluster,3,2)
            npar=random_array(shape,complex)
            ar=tt.slice(npar,cluster=cluster)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainSlice)
