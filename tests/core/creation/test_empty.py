import numpy as np
import ttarray as tt
from ... import DENSE_SHAPE,TINY_SHAPE,LARGE_SHAPE
def check_empty(tta,shape,dt,cluster,chi,typ,dense=True):
    __traceback_hide__ = True
    assert isinstance(tta,typ)
    assert tta.cluster == cluster
    if chi is not None:
        assert tta.chi == chi
    assert tta.shape == shape
    assert tta.dtype == dt
    if dense:
        arb=np.array(tta)
        assert type(arb)==np.ndarray
        assert arb.shape==shape

def test_empty_ttarray():
    for shape,cls in DENSE_SHAPE.items():
        ar=tt.empty(shape)
        ar2=np.empty(shape,like=ar)
        ar3=np.empty_like(ar)
        chi=tuple([1]*(len(cls[0])-1))
        check_empty(ar,shape,float,cls[0],chi,tt.TensorTrainArray)
        check_empty(ar2,shape,float,cls[0],chi,tt.TensorTrainArray)
        check_empty(ar3,shape,float,cls[0],chi,tt.TensorTrainArray)
        for cluster in cls:
            for chi in [1,2,tuple(range(1,len(cluster)))]:
                ar=tt.empty(shape,float,cluster,chi)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                if isinstance(chi,int):
                    chi=tuple([chi]*(len(cluster)-1))
                check_empty(ar,shape,float,cluster,chi,tt.TensorTrainArray)
                check_empty(ar2,shape,float,cluster,chi,tt.TensorTrainArray)
                check_empty(ar3,shape,float,cluster,chi,tt.TensorTrainArray)
def test_empty_ttarray_type():
    for dt in [int,complex,bool]: #float is already tested
        for shape,cls in TINY_SHAPE.items():
            ar=tt.empty(shape,dt)
            ar2=np.empty(shape,dtype=dt,like=ar)
            ar3=np.empty_like(ar)
            chi=tuple([1]*(len(cls[0])-1))
            check_empty(ar,shape,dt,cls[0],chi,tt.TensorTrainArray)
            check_empty(ar2,shape,dt,cls[0],chi,tt.TensorTrainArray)
            check_empty(ar3,shape,dt,cls[0],chi,tt.TensorTrainArray)
            for cluster in cls:
                chi=tuple([1]*(len(cluster)-1))
                ar=tt.empty(shape,dt,cluster)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                check_empty(ar,shape,dt,cluster,chi,tt.TensorTrainArray)
                check_empty(ar2,shape,dt,cluster,chi,tt.TensorTrainArray)
                check_empty(ar3,shape,dt,cluster,chi,tt.TensorTrainArray)

def test_empty_ttslice():
    for shape,cls in DENSE_SHAPE.items():
        shape=(2,)+shape+(3,)
        ar=tt.empty_slice(shape)
        ar2=np.empty(shape,like=ar)
        ar3=np.empty_like(ar)
        chi=tuple([1]*(len(cls[0])-1))
        check_empty(ar,shape,float,cls[0],chi,tt.TensorTrainSlice)
        check_empty(ar2,shape,float,cls[0],chi,tt.TensorTrainSlice)
        check_empty(ar3,shape,float,cls[0],chi,tt.TensorTrainSlice)
        for cluster in cls:
            for chi in [1,2,tuple(range(1,len(cluster)))]:
                ar=tt.empty_slice(shape,float,cluster,chi)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                if isinstance(chi,int):
                    chi=tuple([chi]*(len(cluster)-1))
                check_empty(ar,shape,float,cluster,chi,tt.TensorTrainSlice)
                check_empty(ar2,shape,float,cluster,chi,tt.TensorTrainSlice)
                check_empty(ar3,shape,float,cluster,chi,tt.TensorTrainSlice)

def test_empty_ttslice_type():
    for dt in [int,complex,bool]: #float is already tested
        for shape,cls in TINY_SHAPE.items():
            shape=(2,)+shape+(3,)
            ar=tt.empty_slice(shape,dt)
            ar2=np.empty(shape,dtype=dt,like=ar)
            ar3=np.empty_like(ar)
            chi=tuple([1]*(len(cls[0])-1))
            check_empty(ar,shape,dt,cls[0],chi,tt.TensorTrainSlice)
            check_empty(ar2,shape,dt,cls[0],chi,tt.TensorTrainSlice)
            check_empty(ar3,shape,dt,cls[0],chi,tt.TensorTrainSlice)
            for cluster in cls:
                chi=tuple([1]*(len(cluster)-1))
                ar=tt.empty_slice(shape,dt,cluster)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                check_empty(ar,shape,dt,cluster,chi,tt.TensorTrainSlice)
                check_empty(ar2,shape,dt,cluster,chi,tt.TensorTrainSlice)
                check_empty(ar3,shape,dt,cluster,chi,tt.TensorTrainSlice)

def test_empty_ttslice_large():
    for shape,cls in LARGE_SHAPE.items():
        shape=(2,)+shape+(3,)
        ar=tt.empty_slice(shape)
        # ar2=np.empty(shape,like=ar) #maximum dimension
        ar3=np.empty_like(ar)
        chi=tuple([1]*(len(cls[0])-1))
        check_empty(ar,shape,float,cls[0],chi,tt.TensorTrainSlice,False)
        # check_empty(ar2,shape,float,cls[0],chi,tt.TensorTrainSlice,False)
        check_empty(ar3,shape,float,cls[0],chi,tt.TensorTrainSlice,False)
        for cluster in cls:
            for chi in [1,2,tuple(range(1,len(cluster)))]:
                ar=tt.empty_slice(shape,float,cluster,chi)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                if isinstance(chi,int):
                    chi=tuple([chi]*(len(cluster)-1))
                check_empty(ar,shape,float,cluster,chi,tt.TensorTrainSlice,False)
                # check_empty(ar2,shape,float,cluster,chi,tt.TensorTrainSlice,False)
                check_empty(ar3,shape,float,cluster,chi,tt.TensorTrainSlice,False)

def test_empty_ttarray_large():
    for shape,cls in LARGE_SHAPE.items():
        ar=tt.empty(shape)
        # ar2=np.empty(shape,like=ar) #maximum dimension
        ar3=np.empty_like(ar)
        chi=tuple([1]*(len(cls[0])-1))
        check_empty(ar,shape,float,cls[0],chi,tt.TensorTrainArray,False)
        # check_empty(ar2,shape,float,cls[0],chi,tt.TensorTrainArray,False)
        check_empty(ar3,shape,float,cls[0],chi,tt.TensorTrainArray,False)
        for cluster in cls:
            for chi in [1,2,tuple(range(1,len(cluster)))]:
                ar=tt.empty(shape,float,cluster,chi)
                ar2=tt.empty_like(ar)
                ar3=np.empty_like(ar)
                if isinstance(chi,int):
                    chi=tuple([chi]*(len(cluster)-1))
                check_empty(ar,shape,float,cluster,chi,tt.TensorTrainArray,False)
                # check_empty(ar2,shape,float,cluster,chi,tt.TensorTrainArray,False)
                check_empty(ar3,shape,float,cluster,chi,tt.TensorTrainArray,False)
