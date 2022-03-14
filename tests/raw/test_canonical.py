import pytest
from .. import random_array
from ttarray.raw import is_left_canonical,is_right_canonical,is_canonical
from ttarray.raw import dense_to_ttslice,ttslice_to_dense
from ttarray.raw import canonicalize,singular_values,trivial_decomposition
from ttarray.raw import right_canonicalize,left_canonicalize
from ttarray.raw import right_singular_values,left_singular_values
import numpy.linalg as la

import copy
def test_canonicalize_short(seed_rng,shape):
    shape,cluster=shape
    shape=(2,)+shape+(3,)
    for dt in [int,float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        ttar1=ttar.copy()
        right_canonicalize(ttar1)
        assert is_right_canonical(ttar1)
        ttar2=ttar.copy()
        left_canonicalize(ttar2)
        assert is_left_canonical(ttar2)
        center=len(ttar)//2
        ttar3=ttar.copy()
        canonicalize(ttar3,center)
        assert is_canonical(ttar3,center)
def dense_singular_values(ar):
    pass
def test_svs_short(seed_rng,shape):
    shape,cluster=shape
