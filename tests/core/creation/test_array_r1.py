from ... import random_array,check_ttarray_dense
import numpy as np
from ttarray import frombuffer,fromiter
from ttarray import ones_slice
import pytest
SLICE_PROTOTYPE=ones_slice((2,2,3),int,((2,),),2)

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
def test_ttarray_frombuffer_nocluster(seed_rng,shape_r1):
    shape,cluster=shape_r1
    chi=_calc_chi(cluster)
    for dtype in [np.float32,complex,int]:
        npar=random_array(shape,dtype=dtype)
        buf=npar.tobytes()
        ar=frombuffer(buf,dtype=dtype)
        ar2=np.frombuffer(buf,dtype=dtype,like=ar)
        check_ttarray_dense(ar,npar,cluster,chi)
        check_ttarray_dense(ar2,npar,cluster,chi)

def test_ttarray_fromiter_nocluster(seed_rng,shape_r1):
    shape,cluster=shape_r1
    chi=_calc_chi(cluster)
    for dtype in [np.float32,complex,int]:
        npar=random_array(shape,dtype=dtype)
        iter=list(npar).__iter__()
        ar=fromiter(iter,dtype=dtype)
        iter=list(npar).__iter__()
        ar2=np.fromiter(iter,dtype=dtype,like=ar)
        check_ttarray_dense(ar,npar,cluster,chi)
        check_ttarray_dense(ar2,npar,cluster,chi)
def test_ttslice_frombuffer():
    with pytest.raises(TypeError):
        np.frombuffer(None,like=SLICE_PROTOTYPE)
def test_ttslice_fromiter():
    with pytest.raises(TypeError):
        np.fromiter([1,2,3,4],dtype=int,like=SLICE_PROTOTYPE)
