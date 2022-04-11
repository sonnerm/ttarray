from ttarray.raw import dense_to_ttslice,tensordot,left_canonicalize,right_canonicalize
from .. import random_array,check_raw_ttslice_dense,calc_chi
import numpy.linalg as la
import numpy as np
import pytest
'''
def test_tensordot_r2_r1(seed_rng,shape_cluster_r2):
    shape,cluster=shape_cluster_r2
    shape=(2,)+shape+(3,)
    for dt in [int,float,complex]:
        mar=random_array(shape,dt)
        mttar=dense_to_ttslice(mar,cluster)
        var1=random_array((2,)+(shape[2],)+(3,),dt)
        cl0=tuple((c[0],) for c in cluster)
        cl1=tuple((c[1],) for c in cluster)
        vttar1=dense_to_ttslice(var1,cl1)
        var0=random_array((2,)+(shape[1],)+(3,),dt)
        vttar0=dense_to_ttslice(var0,cl0)
        ttres=tensordot(mttar,vttar1,((1,),(0,)))
        res=np.tensordot(mar,var1,((2,),(1,))).transpose([0,3,1,2,4])
        res=res.reshape((res.shape[0]*res.shape[1],res.shape[2],res.shape[3]*res.shape[4]))
        check_raw_ttslice_dense(ttres,res,cl0,None)
        #normalize_chi (no truncation!)
        left_canonicalize(ttres)
        right_canonicalize(ttres)
        check_raw_ttslice_dense(ttres,np.asarray(res,ttres[0].dtype),cl0,calc_chi(cl0,4,9))

        ttres=tensordot(mttar,vttar0,((0,),(0,)))
        res=np.tensordot(mar,var0,((1,),(1,))).transpose([0,3,1,2,4])
        res=res.reshape((res.shape[0]*res.shape[1],res.shape[2],res.shape[3]*res.shape[4]))
        check_raw_ttslice_dense(ttres,res,cl1,None)
        #normalize_chi (no truncation!)
        left_canonicalize(ttres)
        right_canonicalize(ttres)
        check_raw_ttslice_dense(ttres,np.asarray(res,ttres[0].dtype),cl1,calc_chi(cl1,4,9))


'''
# def test_tensordot_r1_r1(seed_rng,shape_cluster_r1):
#     shape,cluster=shape_cluster_r1
#
# def test_tensordot_r2_r2(seed_rng,shape_cluster_r2):
#     shape,cluster=shape_cluster_r2
