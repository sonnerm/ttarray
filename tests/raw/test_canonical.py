import pytest
from .. import random_array,check_raw_ttslice_dense,calc_chi, DENSE_SHAPE
import ttarray.raw as raw

import numpy.linalg as la
import functools

import copy

def test_canonical_shift_short(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        cluster=cls[0]
        chi=calc_chi(cluster,2,3)
        shape=(2,)+shape+(3,)
        for dt in [float,complex]:
            ar=random_array(shape,dt)
            ttar=raw.dense_to_ttslice(ar,cluster,raw.trivial_decomposition)
            center=len(ttar)//2
            ttar2=ttar.copy()
            raw.left_canonicalize(ttar)
            check_raw_ttslice_dense(ttar,ar,cluster,chi)
            assert raw.is_left_canonical(ttar)
            assert raw.is_canonical(ttar,raw.find_orthogonality_center(ttar))
            assert raw.is_canonical(ttar,-1)
            raw.shift_orthogonality_center(ttar,-1,0)
            check_raw_ttslice_dense(ttar,ar,cluster,chi)
            assert raw.is_right_canonical(ttar)
            assert raw.is_canonical(ttar,raw.find_orthogonality_center(ttar))
            assert raw.is_canonical(ttar,0)
            raw.shift_orthogonality_center(ttar,0,0)
            check_raw_ttslice_dense(ttar,ar,cluster,chi)
            assert raw.is_right_canonical(ttar)
            assert raw.is_canonical(ttar,raw.find_orthogonality_center(ttar))
            raw.shift_orthogonality_center(ttar,0,center)
            check_raw_ttslice_dense(ttar,ar,cluster,chi)
            assert raw.is_canonical(ttar,center)
            assert raw.is_canonical(ttar2,raw.find_orthogonality_center(ttar2))
            raw.canonicalize(ttar2,center)
            check_raw_ttslice_dense(ttar2,ar,cluster,chi)
            assert raw.is_canonical(ttar2,center)
            assert raw.is_canonical(ttar2,raw.find_orthogonality_center(ttar2))
            raw.shift_orthogonality_center(ttar2,center,center)
            check_raw_ttslice_dense(ttar2,ar,cluster,chi)
            assert raw.is_canonical(ttar2,center)
            assert raw.is_canonical(ttar2,raw.find_orthogonality_center(ttar2))
