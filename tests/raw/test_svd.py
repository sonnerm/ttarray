import numpy as np
import numpy.linalg as la
from ttarray.raw import is_left_canonical,is_right_canonical,is_canonical
from ttarray.raw import dense_to_ttslice,ttslice_to_dense
from ttarray.raw import singular_values,trivial_decomposition
from ttarray.raw import right_canonicalize,left_canonicalize
from ttarray.raw import right_singular_values,left_singular_values
from ttarray.raw import shift_orthogonality_center_with_singular_values
from ttarray.raw import left_truncate_svd,right_truncate_svd
import pytest
import functools
from .. import random_array,check_raw_ttslice_dense,calc_chi
def dense_singular_values(ar,cluster):
    cl=(1,)*len(ar.shape[1:-1])
    arsh=ar.shape
    svds=[]
    for c in cluster:
        cl=tuple(i*ic for i,ic in zip(cl,c))
        ari=ar.reshape((arsh[0],)+sum(((cc,i//cc) for i,cc in zip(arsh[1:-1],cl)),())+(arsh[-1],))
        tp=(0,)+tuple(range(1,2*len(cl)+1,2))+tuple(range(2,2*len(cl)+1,2))+(-1,)
        ari=ari.transpose(tp)
        ari=ari.reshape((arsh[0]*_product(cl),-1))
        svds.append(la.svd(ari,compute_uv=False))
    return svds[:-1]

def _product(seq):
    return functools.reduce(lambda x,y:x*y, seq,1)
def test_shift_with_sv_short(seed_rng,shape):
    shape,cluster=shape
    chi=calc_chi(cluster,2,3)
    shape=(2,)+shape+(3,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        svc=dense_singular_values(ar,cluster)
        center=len(ttar)//2
        ttar1=ttar.copy()
        right_canonicalize(ttar1)
        assert is_right_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        svs1=right_singular_values(ttar1,inplace=True)
        for ss,sc in zip(svs1,svc):
            assert ss==pytest.approx(sc)
        assert is_left_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        shift_orthogonality_center_with_singular_values(ttar1,-1,-1,svs1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_left_canonical(ttar1)
        shift_orthogonality_center_with_singular_values(ttar1,-1,0,svs1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_right_canonical(ttar1)
        svs2=singular_values(ttar1,0)
        for ss,sc in zip(svs2,svc):
            assert ss==pytest.approx(sc)
        shift_orthogonality_center_with_singular_values(ttar1,0,center,svs2)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        assert is_canonical(ttar1,center)
        svs3=singular_values(ttar1,center)
        for ss,sc in zip(svs3,svc):
            assert ss==pytest.approx(sc)
        left_canonicalize(ttar1)
        assert is_left_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)
        svs4=left_singular_values(ttar1,inplace=True)
        for ss,sc in zip(svs4,svc):
            assert ss==pytest.approx(sc)
        assert is_right_canonical(ttar1)
        check_raw_ttslice_dense(ttar1,ar,cluster,chi)

def test_singular_values(seed_rng,shape):
    shape,cluster=shape
    chi=calc_chi(cluster,1,1)
    shape=(1,)+shape+(1,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        svc=dense_singular_values(ar,cluster)
        right_canonicalize(ttar)
        sv=right_singular_values(ttar)
        for ss,sc in zip(sv,svc):
            assert ss==pytest.approx(sc)
def test_truncate(seed_rng,shape):
    shape,cluster=shape
    chi=calc_chi(cluster,2,3)
    shape=(2,)+shape+(3,)
    for dt in [float,complex]:
        ar=random_array(shape,dt)
        ttar=dense_to_ttslice(ar,cluster,trivial_decomposition)
        svc=dense_singular_values(ar,cluster)
        ttar2=ttar.copy()
        right_canonicalize(ttar)
        sv=right_truncate_svd(ttar,chi_max=20,cutoff=0.0)
        for ss,sc in zip(sv,svc):
            assert ss==pytest.approx(sc[:20],rel=1e-1)
        assert is_left_canonical(ttar)
        sv2=left_singular_values(ttar)
        for ss,sc in zip(sv,sv2):
            assert ss==pytest.approx(sc[:20],rel=2e-1,abs=1e-1)
            assert sc[20:]==pytest.approx(0.0)
        svc2=dense_singular_values(ttslice_to_dense(ttar),cluster)
        for ss,sc in zip(sv2,svc2):
            assert ss==pytest.approx(sc[:20])

        left_canonicalize(ttar2)
        sv=left_truncate_svd(ttar2,chi_max=None,cutoff=0.2)
        for ss,sc in zip(sv,svc):
            assert ss==pytest.approx(sc[sc>0.2],rel=1e-1)
        assert is_right_canonical(ttar2)
        sv2=right_singular_values(ttar2)
        for ss,sc in zip(sv,sv2):
            assert ss==pytest.approx(sc[sc>0.2],rel=1e-1)
        svc2=dense_singular_values(ttslice_to_dense(ttar2),cluster)
        for ss,sc in zip(sv2,svc2):
            assert ss==pytest.approx(sc[sc>0.2])
