import numpy as np
import ttarray as tt
from ... import check_dense,random_array
from ... import DENSE_SHAPE
import pytest
SLICE_PROTOTYPE=tt.ones_slice((2,2,3),int,((2,),),2)

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
def test_ttarray_frombuffer(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        if len(shape)!=1:
            continue
        cluster=cls[0]
        chi=_calc_chi(cluster)
        npar=random_array(shape,dtype=np.float32)
        buf=npar.tobytes()
        ar=tt.frombuffer(buf,dtype=np.float32)
        ar2=np.frombuffer(buf,dtype=np.float32,like=ar)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            chi=_calc_chi(cluster)
            ar=tt.frombuffer(buf,dtype=np.float32,cluster=cluster)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)

def test_ttarray_fromiter(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        if len(shape)!=1:
            continue
        cluster=cls[0]
        chi=_calc_chi(cluster)
        npar=random_array(shape,dtype=np.int32)
        ar=tt.fromiter(iter(npar),dtype=np.int32)
        ar2=np.fromiter(iter(npar),dtype=np.int32,like=ar)
        check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)
        check_dense(ar2,npar,cluster,chi,tt.TensorTrainArray)
        for cluster in cls:
            chi=_calc_chi(cluster)
            ar=tt.fromiter(iter(npar),dtype=np.int32,cluster=cluster)
            check_dense(ar,npar,cluster,chi,tt.TensorTrainArray)

def test_ttslice_frombuffer():
    with pytest.raises(TypeError):
        np.frombuffer(None,like=SLICE_PROTOTYPE)
def test_ttslice_fromiter():
    with pytest.raises(TypeError):
        np.fromiter([1,2,3,4],dtype=int,like=SLICE_PROTOTYPE)
