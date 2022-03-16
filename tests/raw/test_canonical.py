import pytest
from .. import random_array,check_raw_ttslice_dense
from ttarray.raw import is_left_canonical,is_right_canonical,is_canonical
from ttarray.raw import dense_to_ttslice,ttslice_to_dense
from ttarray.raw import canonicalize,singular_values,trivial_decomposition
from ttarray.raw import right_canonicalize,left_canonicalize
from ttarray.raw import right_singular_values,left_singular_values
from ttarray.raw import shift_orthogonality_center,shift_orthogonality_center_with_singular_values
import numpy.linalg as la
import functools

import copy
def dense_singular_values(ar,cluster):
    cl=(1,)*len(ar.shape[1:-1])
    arsh=ar.shape
    svds=[]
    for c in (((1,)*len(cl),)+cluster):
        cl=tuple(i*ic for i,ic in zip(cl,c))
        ari=ar.reshape((arsh[0],)+sum(((cc,i//cc) for i,cc in zip(arsh[1:-1],cl)),())+(arsh[-1],))
        tp=(0,)+tuple(range(1,2*len(cl)+1,2))+tuple(range(2,2*len(cl)+1,2))+(-1,)
        ari=ari.transpose(tp)
        ari=ari.reshape((arsh[0]*_product(cl),-1))
        svds.append(la.svd(ari,compute_uv=False))
    return svds

def _product(seq):
    return functools.reduce(lambda x,y:x*y, seq,1)
def _calc_chi(cluster,lefti=1,righti=1):
    left,right=[lefti],[righti]
    for c in cluster:
        left.append(left[-1]*_product(c))
    for c in cluster[::-1]:
        right.append(right[-1]*_product(c))
    return tuple(min(l,r) for l,r in zip(left[1:-1],right[1:-1][::-1]))
def test_canonicalize_svs_short(seed_rng,shape):
    shape,cluster=shape
    chi=_calc_chi(cluster,2,3)
    shape=(2,)+shape+(3,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        svc=dense_singular_values(ar,cluster)
        center=len(ttar)//2
        ttar1=ttar.copy()
        right_canonicalize(ttar1)
        assert is_right_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        svs1=right_singular_values(ttar1,inplace=True)
        for ss,sc in zip(svs1,svc):
            assert ss==pytest.approx(sc)
        assert is_left_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        shift_orthogonality_center_with_singular_values(ttar1,-1,-1,svs1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_left_canonical(ttar1)
        shift_orthogonality_center_with_singular_values(ttar1,-1,0,svs1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_right_canonical(ttar1)
        svs2=singular_values(ttar1,0)
        for ss,sc in zip(svs2,svc):
            assert ss==pytest.approx(sc)
        shift_orthogonality_center_with_singular_values(ttar1,0,center,svs2)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_canonical(ttar1,center)
        svs3=singular_values(ttar1,center)
        for ss,sc in zip(svs3,svc):
            assert ss==pytest.approx(sc)
        left_canonicalize(ttar1)
        assert is_left_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        svs4=left_singular_values(ttar1,inplace=True)
        for ss,sc in zip(svs4,svc):
            assert ss==pytest.approx(sc)
        assert is_right_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)

def test_canonical_shift_short(seed_rng,shape):
    shape,cluster=shape
    chi=_calc_chi(cluster,2,3)
    shape=(2,)+shape+(3,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        center=len(ttar)//2
        ttar2=ttar.copy()
        left_canonicalize(ttar)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_left_canonical(ttar)
        assert is_canonical(ttar,-1)
        shift_orthogonality_center(ttar,-1,0)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_right_canonical(ttar)
        assert is_canonical(ttar,0)
        shift_orthogonality_center(ttar,0,0)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_right_canonical(ttar)
        shift_orthogonality_center(ttar,0,center)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_canonical(ttar,center)
        canonicalize(ttar2,center)
        check_raw_ttslice_dense(ttar2,ar,cluster,chi)
        assert is_canonical(ttar2,center)
        shift_orthogonality_center(ttar2,center,center)
        check_raw_ttslice_dense(ttar2,ar,cluster,chi)
        assert is_canonical(ttar2,center)
def test_singular_values(seed_rng,shape):
    shape,cluster=shape
    chi=_calc_chi(cluster,1,1)
    shape=(1,)+shape+(1,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        svc=dense_singular_values(ar,cluster)
        right_canonicalize(ttar)
        sv=right_singular_values(ttar)
        for ss,sc in zip(sv,svc):
            assert ss==pytest.approx(sc)
