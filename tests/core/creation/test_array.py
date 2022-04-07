import numpy as np
from ... import check_ttarray_dense,check_ttslice_dense,random_array
from ttarray import array,asarray,asanyarray,asanyslice,asslice,slice
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
def test_ttarray_ndarray(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    chi=_calc_chi(cluster)
    for dt in [complex,np.float64,np.float32,int]:
        npar=random_array(shape,dt)
        ar=array(npar,cluster=cluster)
        check_ttarray_dense(ar,npar,cluster,chi)

def test_ttslice_ndarray(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=tuple([2]+list(shape)+[3])
    chi=_calc_chi(cluster,2,3)
    for dt in [complex,np.float64,np.float32,int]:
        npar=random_array(shape,dt)
        ar=slice(npar,cluster=cluster)
        check_ttslice_dense(ar,npar,cluster,chi)

def test_ttarray_ndarray_nocluster(shape):
    shape,cluster=shape
    chi=_calc_chi(cluster)
    for dt in [complex,np.float64,np.float32,int]:
        npar=random_array(shape,dt)
        ar=array(npar)
        ar2=np.array(npar,like=ar)
        ar3=np.asarray(npar,like=ar)
        ar4=np.asanyarray(npar,like=ar)
        # ar5=np.asfarray(ar)
        # ar6=np.asfarray(ar,dtype=dt)
        check_ttarray_dense(ar,npar,cluster,chi)
        check_ttarray_dense(ar2,npar,cluster,chi)
        check_ttarray_dense(ar3,npar,cluster,chi)
        check_ttarray_dense(ar4,npar,cluster,chi)
        # check_ttarray_dense(ar5,np.asfarray(npar),cluster,chi)
        # check_ttarray_dense(ar6,np.asfarray(npar,dtype=dt),cluster,chi)
        ar=array(npar,dtype=complex)
        ar2=np.array(npar,dtype=complex,like=ar)
        ar3=np.asarray(npar,dtype=complex,like=ar)
        ar4=np.asanyarray(npar,dtype=complex,like=ar)
        # ar5=np.asfarray(ar,dtype=complex)
        npar2=np.array(npar,dtype=complex)
        check_ttarray_dense(ar,npar2,cluster,chi)
        check_ttarray_dense(ar2,npar2,cluster,chi)
        check_ttarray_dense(ar3,npar2,cluster,chi)
        check_ttarray_dense(ar4,npar2,cluster,chi)
        # check_ttarray_dense(ar5,npar2,cluster,chi)
def test_ttslice_ndarray_nocluster(shape):
    shape,cluster=shape
    shape=tuple([3]+list(shape)+[2])
    chi=_calc_chi(cluster,3,2)
    for dt in [complex,np.float64,np.float32,int]:
        npar=random_array(shape,dt)
        ar=slice(npar)
        ar2=np.array(npar,like=ar)
        ar3=np.asarray(npar,like=ar)
        ar4=np.asanyarray(npar,like=ar)
        # ar5=np.asfarray(ar)
        # ar6=np.asfarray(ar,dtype=dt)
        check_ttslice_dense(ar,npar,cluster,chi)
        check_ttslice_dense(ar2,npar,cluster,chi)
        check_ttslice_dense(ar3,npar,cluster,chi)
        check_ttslice_dense(ar4,npar,cluster,chi)
        # check_ttarray_dense(ar5,np.asfarray(npar),cluster,chi)
        # check_ttarray_dense(ar6,np.asfarray(npar,dtype=dt),cluster,chi)
        ar=slice(npar,dtype=complex)
        ar2=np.array(npar,dtype=complex,like=ar)
        ar3=np.asarray(npar,dtype=complex,like=ar)
        ar4=np.asanyarray(npar,dtype=complex,like=ar)
        # ar5=np.asfarray(ar,dtype=complex)
        npar2=np.array(npar,dtype=complex)
        check_ttslice_dense(ar,npar2,cluster,chi)
        check_ttslice_dense(ar2,npar2,cluster,chi)
        check_ttslice_dense(ar3,npar2,cluster,chi)
        check_ttslice_dense(ar4,npar2,cluster,chi)
        # check_ttarray_dense(ar5,npar2,cluster,chi)

# def test_ttarray_ttarray(shape_cluster):
#     pass
#
# def test_ttarray_ttslice(shape_cluster):
#     pass
#
# def test_ttslice_ttslice(shape_cluster):
#     pass
#
# def test_ttslice_ttarray(shape_cluster):
#     pass
