import ttarray.raw as raw
from .. import random_array,check_raw_ttslice_dense,calc_chi,DENSE_SHAPE
import numpy.linalg as la
import pytest
import copy

import itertools
# SHAPE_RECLUSTER=[
# ((2,3),[((),),((),(),()),((),())]),
# ((2,24,3),[((24,),),((1,),(24,)),((3,),(4,),(2,)),((2,),(2,),(2,),(3,)),((24,),(1,),(1,)),((3,),(2,),(2,),(1,),(2,))]),
# ((2,64,3),[((2,),(32,)),((2,),(2,),(2,),(2,),(2,),(2,)),((4,),(4,),(4,)),((2,),(4,),(4,),(2,)),((4,),(4,),(4,)),((2,),(2,),(2,),(2,),(2,),(2,))]),
# ((2,64,24,1),[((2,2),(32,12)),((2,1),(2,1),(2,1),(2,24),(2,1),(2,1)),((4,4),(4,2),(4,3)),((2,3),(1,8),(4,1),(4,1),(2,1)),((4,2),(4,2),(4,6)),((2,2),(2,2),(2,2),(2,3),(2,1),(2,1))]),
# ]
#
# @pytest.fixture(params=SHAPE_RECLUSTER)
# def shape_recluster(request):
#     return request.param
def test_recluster(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        ar=random_array(shape,float)
        for c1 in cls:
            ttar=raw.dense_to_ttslice(ar,c1,la.qr)
            check_raw_ttslice_dense(ttar,ar,c1,calc_chi(c1,2,3))
            for c2 in cls:
                ttar2=raw.recluster(copy.copy(ttar),c2,raw.trivial_decomposition)
                check_raw_ttslice_dense(ttar2,ar,c2,None)
