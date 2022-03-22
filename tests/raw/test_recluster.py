from ttarray.raw import dense_to_ttslice,recluster
from .. import random_array,check_raw_ttslice_dense,calc_chi
import numpy.linalg as la
import pytest

import itertools
SHAPE_RECLUSTER=[
((2,3),[((),),((),(),()),((),())]),
((2,24,3),[((24,),),((1,),(24,)),((3,),(4,),(2,)),((2,),(2,),(2,),(3,)),((24,),(1,),(1,)),((3,),(2,),(2,),(1,),(2,))]),
((2,64,3),[((2,),(32,)),((2,),(2,),(2,),(2,),(2,),(2,)),((4,),(4,),(4,)),((2,),(4,),(4,),(2,)),((4,),(4,),(4,)),((2,),(2,),(2,),(2,),(2,),(2,))]),
((2,64,24,1),[((2,2),(32,12)),((2,1),(2,1),(2,1),(2,24),(2,1),(2,1)),((4,4),(4,2),(4,3)),((2,3),(1,8),(4,1),(4,1),(2,1)),((4,2),(4,2),(4,6)),((2,2),(2,2),(2,2),(2,3),(2,1),(2,1))]),
]

@pytest.fixture(params=SHAPE_RECLUSTER)
def shape_recluster(request):
    return request.param
def test_recluster_float(seed_rng,shape_recluster):
    shape,clusters=shape_recluster
    ar=random_array(shape,float)
    ttar=dense_to_ttslice(ar,clusters[0],la.qr)
    for c in clusters:
        ttar=recluster(ttar,c)
        check_raw_ttslice_dense(ttar,ar,c,None)
        # check_raw_ttslice_dense(ttar,ar,c,calc_chi(c,2,3))
