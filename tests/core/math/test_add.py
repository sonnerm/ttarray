import numpy as np
import ttarray as tt
from ... import check_dense,random_array, random_ttarray, random_ttslice
from ... import DENSE_SHAPE,LARGE_SHAPE,TINY_SHAPE
import pytest
def test_add_ttarray_ttarray_dense(seed_rng):
    # for shape,cls in DENSE_SHAPE.items():
    #     ar1=random_array(shape,dt)
    #     ar2=random_array(shape,dt)
    #     ars=ar1+ar2
    #     ttar1=tt.fromdense(ar1,cluster)
    #     ttar2=tt.fromdense(ar2,cluster)
    #     ttars1=ttar1+ttar2
    #     ttars2=np.add(ttar1,ttar2)
    #     ttars3=tt.add(ttar1,ttar2)
    #     ttars4=ttar1.copy()
    #     ttars5=ttar2.copy()
    #     ttars4+=ttar2
    #     np.add(ttar1,ttars5,out=ttars5)
    #     check_ttarray_dense(ttars1,ars)
    #     check_ttarray_dense(ttars2,ars)
    #     check_ttarray_dense(ttars3,ars)
    #     check_ttarray_dense(ttars4,ars)
    #     check_ttarray_dense(ttars5,ars)
    pass
