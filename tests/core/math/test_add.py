import numpy as np
import ttarray as tt
from ... import check_dense,random_array, random_ttarray, random_ttslice
from ... import DENSE_SHAPE,LARGE_SHAPE,TINY_SHAPE
import pytest
def test_add_ttarray_ttarray_dense(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        ar1=random_array(shape,float)
        ar2=random_array(shape,float)
        ars=ar1+ar2
        for i,c1 in enumerate(cls):
            c2=cls[(i+1)%len(cls)]
            ttar1=tt.array(ar1,cluster=c1)
            ttar2=tt.array(ar2,cluster=c2)
            ttars1=ttar1+ttar2
            ttars2=ttar2+ttar1
            ttars3=np.add(ttar1,ttar2)
            ttars4=tt.add(ttar2,ttar1)
            # ttars4=ttar1.copy()
            # ttars5=ttar2.copy()
            # ttars4+=ttar2
            # np.add(ttar1,ttars5,out=ttars5)
            check_dense(ttars1,ars,c2,None,tt.TensorTrainArray)
            check_dense(ttars2,ars,c1,None,tt.TensorTrainArray)
            check_dense(ttars3,ars,c2,None,tt.TensorTrainArray)
            check_dense(ttars4,ars,c1,None,tt.TensorTrainArray)
            check_dense(ttar1,ar1,c1,None,tt.TensorTrainArray)
            check_dense(ttar2,ar2,c2,None,tt.TensorTrainArray) #stay unmodified
            # check_dense(ttars4,ars,c2,None,tt.TensorTrainArray)
            # check_dense(ttars5,ars,c2,None,tt.TensorTrainArray)

def test_add_ttarray_scalar_dense(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        ar=random_array(shape,float)
        scal=np.random.random()
        ars=scal+ar
        for cluster in cls:
            ttar=tt.array(ar,cluster=cluster)
            ttars1=scal+ttar
            ttars2=ttar+scal
            ttars3=np.add(scal,ttar)
            ttars4=tt.add(ttar,scal)
            check_dense(ttars1,ars,cluster,None,tt.TensorTrainArray)
            check_dense(ttars2,ars,cluster,None,tt.TensorTrainArray)
            check_dense(ttars3,ars,cluster,None,tt.TensorTrainArray)
            check_dense(ttars4,ars,cluster,None,tt.TensorTrainArray)
