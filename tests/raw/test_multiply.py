import pytest
from .. import random_array
from ttarray.raw import trivial_decomposition,dense_to_ttslice,ttslice_to_dense
from ttarray.raw import multiply_ttslice
def test_multiply_short(seed_rng,shape_cluster):
    shape,cluster=shape_cluster
    shape=(2,)+shape+(3,)
    for dt in [int,float,complex]:
        ar1=random_array(shape,dt)
        ar2=random_array(shape,dt)
        ttar1=dense_to_ttslice(ar1,cluster,trivial_decomposition)
        ttar2=dense_to_ttslice(ar2,cluster,trivial_decomposition)
        res=multiply_ttslice(ttar1,ttar2)
        resa=ttslice_to_dense(res)
        resa==pytest.approx(ar1*ar2)
        assert resa.dtype==dt
