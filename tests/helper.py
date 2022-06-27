import numpy as np
import pytest
import ttarray as tt
import functools
TINY_SHAPE={}
TINY_SHAPE[()]=[((),),((),()),((),(),()),((),(),(),())]
TINY_SHAPE[(1,)]=[((1,),),((1,),(1,)),((1,),(1,),(1,)),((1,),(1,),(1,),(1,))]
TINY_SHAPE[(1,1)]=[((1,1),),((1,1),(1,1)),((1,1),(1,1),(1,1),(1,1))]
TINY_SHAPE[(1,1,1)]=[((1,1,1),),((1,1,1),(1,1,1),(1,1,1),(1,1,1))]
TINY_SHAPE[(2,)]=[((2,),),((2,),(1,)),((1,),(2,),(1,)),((1,),(2,))]
DENSE_SHAPE=dict(TINY_SHAPE)
DENSE_SHAPE[(16,)]=[((2,),(2,),(2,),(2,))]
DENSE_SHAPE[(24,)]=[((3,),(2,),(2,),(2,))]
DENSE_SHAPE[(30,)]=[((5,),(3,),(2,))]
DENSE_SHAPE[(128,)]=[((2,),(2,),(2,),(2,),(2,),(2,),(2,))]

LARGE_SHAPE={}
LARGE_SHAPE[(2**90,)]=[((2,),)*90,((4,),)*45,((4,),(2,))*30,((2,),(4,))*30]
LARGE_SHAPE[(2**90,2**90)]=[((2,2),)*90,((4,4),)*45,((4,2),(2,4))*30,((2,2),(4,4))*30,((2,4),(4,2))*30]
def check_dense(tta,ar,cluster,chi,typ):
    __traceback_hide__ = True
    assert isinstance(tta,typ)
    assert tta.cluster == cluster
    if chi is not None:
        assert tta.chi == chi
    assert tta.shape == ar.shape
    assert tta.dtype == ar.dtype
    arb=np.array(tta)
    assert type(arb)==np.ndarray
    assert arb == pytest.approx(ar)

def check_constant(tta,val,shape,cluster,chi,typ):
    __traceback_hide__ = True
    assert isinstance(tta,typ)
    assert tta.cluster == cluster
    if chi is not None:
        assert tta.chi == chi
    assert tta.shape == shape
    assert tta.dtype == np.dtype(type(val))
    #check actual value
    #assert tt.sum(tt.asfarray(tta))==pytest.approx(functools.reduce(lambda x,y:x*y,tta.shape)*val)
    #assert tta.conj()@tta==functools.reduce(lambda x,y:x*y,tta.shape)*val**2

def check_raw_ttslice_dense(tta,ar,cluster,chi):
    __traceback_hide__ = True
    check_dense(tt.TensorTrainSlice.frommatrices(tta),ar,cluster,chi,tt.TensorTrainSlice)
def random_array(shape,dtype=complex):
    if np.dtype(dtype).kind=="i":
        return np.random.randint(-10000000,10000000,size=shape,dtype=dtype)
    elif np.dtype(dtype).kind=="f":
        return np.array(np.random.randn(*shape)-1,dtype=dtype)
    elif np.dtype(dtype).kind=="c":
        return np.array(np.random.randn(*shape)+1.0j*np.random.randn(*shape),dtype=dtype)
    else:
        raise NotImplementedError("unknown dtype")
def random_ttarray(shape=None,dtype=complex,cluster=None,chi=None):
    if shape==None:
        shape=(24,35)
        cluster=((2,1),(2,7),(1,1),(6,5),(1,1))
    if cluster==None:
        cluster=ttarray.raw.get_balanced_cluster(shape)
    if chi==None:
        chi=tuple(np.arange(len(cluster)-1)+3)
    chi=(1,)+chi+(1,)
    Ms=[random_array((chl,)+clu+(chr,),dtype) for chl,clu,chr in zip(chi[:-1],cluster,chi[1:])]
    return tt.frommatrices(Ms)


def random_ttslice(shape=None,dtype=complex,cluster=None,chi=None):
    if shape==None:
        shape=(2,12,3)
        cluster=((2,),(3,),(1,),(2,))
    if cluster==None:
        cluster=ttarray.raw.get_balanced_cluster(shape[1:-1])
    if chi==None:
        chi=tuple(np.arange(len(cluster)-1)+3)
    chi=(shape[0],)+chi+(shape[-1],)
    Ms=[random_array((chl,)+clu+(chr,),dtype) for chl,clu,chr in zip(chi[:-1],cluster,chi[1:])]
    return tt.frommatrices_slice(Ms)
def _product(seq):
    return functools.reduce(lambda x,y:x*y, seq,1)
def calc_chi(cluster,lefti=1,righti=1):
    left,right=[lefti],[righti]
    for c in cluster:
        left.append(left[-1]*_product(c))
    for c in cluster[::-1]:
        right.append(right[-1]*_product(c))
    return tuple(min(l,r) for l,r in zip(left[1:-1],right[1:-1][::-1]))
