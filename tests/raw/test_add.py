import pytest
from .. import random_array,DENSE_SHAPE
import ttarray.raw as raw
def test_add_short(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        for cluster in cls:
            ar1=random_array(shape,float)
            ar2=random_array(shape,float)
            ttar1=raw.dense_to_ttslice(ar1,cluster,raw.trivial_decomposition)
            ttar2=raw.dense_to_ttslice(ar2,cluster,raw.trivial_decomposition)
            res=raw.add_ttslice(ttar1,ttar2)
            resa=raw.ttslice_to_dense(res)
            resa==pytest.approx(ar1+ar2)
