import pytest
from .. import random_array,check_raw_ttslice_dense,calc_chi
from ttarray.raw import is_left_canonical,is_right_canonical,is_canonical
from ttarray.raw import dense_to_ttslice,ttslice_to_dense
from ttarray.raw import canonicalize,trivial_decomposition
from ttarray.raw import right_canonicalize,left_canonicalize
from ttarray.raw import shift_orthogonality_center
from ttarray.raw import find_orthogonality_center
import numpy.linalg as la
import functools

import copy

def test_canonical_shift_short(seed_rng,shape):
    shape,cluster=shape
    chi=calc_chi(cluster,2,3)
    shape=(2,)+shape+(3,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        center=len(ttar)//2
        ttar2=ttar.copy()
        left_canonicalize(ttar)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_left_canonical(ttar)
        assert is_canonical(ttar,find_orthogonality_center(ttar))
        assert is_canonical(ttar,-1)
        shift_orthogonality_center(ttar,-1,0)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_right_canonical(ttar)
        assert is_canonical(ttar,find_orthogonality_center(ttar))
        assert is_canonical(ttar,0)
        shift_orthogonality_center(ttar,0,0)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_right_canonical(ttar)
        assert is_canonical(ttar,find_orthogonality_center(ttar))
        shift_orthogonality_center(ttar,0,center)
        check_raw_ttslice_dense(ttar,ar,cluster,chi)
        assert is_canonical(ttar,center)
        assert is_canonical(ttar2,find_orthogonality_center(ttar2))
        canonicalize(ttar2,center)
        check_raw_ttslice_dense(ttar2,ar,cluster,chi)
        assert is_canonical(ttar2,center)
        assert is_canonical(ttar2,find_orthogonality_center(ttar2))
        shift_orthogonality_center(ttar2,center,center)
        check_raw_ttslice_dense(ttar2,ar,cluster,chi)
        assert is_canonical(ttar2,center)
        assert is_canonical(ttar2,find_orthogonality_center(ttar2))
