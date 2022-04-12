import numpy as np
import ttarray as tt
from ... import check_dense,random_array, random_ttarray, random_ttslice
from ... import DENSE_SHAPE,LARGE_SHAPE,TINY_SHAPE
import pytest
def test_add_ttarray_ttarray_dense(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        ar1=random_array(shape,float)
        ar2=random_array(shape,float)
        ars=ar1*ar2
        for i,c1 in enumerate(cls):
            c2=cls[(i+1)%len(cls)]
            ttar1=tt.array(ar1,cluster=c1)
            ttar2=tt.array(ar2,cluster=c2)
            ttars1=ttar1*ttar2
            ttars2=ttar2*ttar1
            ttars3=np.multiply(ttar1,ttar2)
            ttars4=tt.multiply(ttar2,ttar1)
            check_dense(ttars1,ars,c2,None,tt.TensorTrainArray)
            check_dense(ttars2,ars,c1,None,tt.TensorTrainArray)
            check_dense(ttars3,ars,c2,None,tt.TensorTrainArray)
            check_dense(ttars4,ars,c1,None,tt.TensorTrainArray)
