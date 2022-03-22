import pytest
import numpy as np
import hashlib
SHAPE_CLUSTER_R0=[((),((),)),((),((),())),((),((),(),(),()))]
SHAPE_CLUSTER_R1=[((1,),((1,),)),((1,),((1,),(1,))),((1,),((1,),(1,),(1,),(1,)))]
SHAPE_CLUSTER_R1+=[((16,),((16,),)),((16,),((4,),(4,))),((16,),((2,),(2,),(2,),(2,)))]
SHAPE_CLUSTER_R1+=[((16,),((8,),(2,))),((16,),((1,),(2,),(4,),(2,))),((16,),((2,),(2,),(4,)))]
SHAPE_CLUSTER_R1+=[((24,),((2,),(3,),(1,),(4,)))]
SHAPE_CLUSTER_R1+=[((2,),((2,),)),((2,),((1,),(2,),(1,))),((4,),((4,),)),((4,),((2,),(2,)))]

SHAPE_CLUSTER_R2=[((1,1),((1,1),)),((1,1),((1,1),(1,1))),((1,1),((1,1),(1,1),(1,1),(1,1)))]
SHAPE_CLUSTER_R2+=[((2,2),((2,2),)),((2,2),((2,1),(1,2))),((2,2),((1,1),(2,1),(1,2),(1,1)))]
SHAPE_CLUSTER_R2+=[((4,4),((2,2),(2,2))),((8,4),((2,2),(2,1),(2,2))),((4,8),((1,2),(2,2),(2,2)))]

SHAPE_CLUSTER_R3=[((1,1,1),((1,1,1),)),((1,1,1),((1,1,1),(1,1,1))),((1,1,1),((1,1,1),(1,1,1),(1,1,1),(1,1,1)))]

SHAPE_R0=[((),((),))]
SHAPE_R1=[((1,),((1,),)),((2,),((2,),)),((4,),((2,),(2,))),((16,),((2,),(2,),(2,),(2,)))]
SHAPE_R1+=[((24,),((3,),(2,),(2,),(2,))),((30,),((5,),(3,),(2,))),((128,),((2,),(2,),(2,),(2,),(2,),(2,),(2,)))]
SHAPE_R2=[((1,1),((1,1),)),((2,2),((2,2),)),((4,4),((2,2),(2,2))),((8,8),((2,2),(2,2),(2,2)))]
SHAPE_R2+=[((8,25),((2,5),(2,5),(2,1))),((24,8),((3,2),(2,2),(2,2),(2,1)))]
SHAPE_R2+=[((8,16),((2,2),(2,2),(2,2),(1,2)))]
SHAPE_R3=[((1,1,1),((1,1,1),))]
SHAPE_CLUSTER=SHAPE_CLUSTER_R0+SHAPE_CLUSTER_R1+SHAPE_CLUSTER_R2+SHAPE_CLUSTER_R3
SHAPE=SHAPE_R0+SHAPE_R1+SHAPE_R2+SHAPE_R3
# LARGE_SHAPE=[(2**100),(2,)*100,(2**50,3**50),(2,3)*50]
# LARGE_SHAPE_CLUSTER=
@pytest.fixture(params=SHAPE_CLUSTER_R1+SHAPE_R1)
def shape_cluster_r1(request):
    return request.param

@pytest.fixture(params=SHAPE_CLUSTER_R2+SHAPE_R2)
def shape_cluster_r2(request):
    return request.param

@pytest.fixture(params=SHAPE_CLUSTER)
def shape_cluster(request):
    return request.param

@pytest.fixture(params=SHAPE)
def shape(request):
    return request.param

@pytest.fixture(params=SHAPE_R1)
def shape_r1(request):
    return request.param

@pytest.fixture(scope="function")
def seed_rng(request):
    '''
        Seeds the random number generator of numpy to a predictable value which
        is different for each test case
    '''
    stri=request.node.name+"_ttarray_test"
    np.random.seed(int.from_bytes(hashlib.md5(stri.encode('utf-8')).digest(),"big")%2**32)
    return None
