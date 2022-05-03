import numpy as np
import ttarray as tt
from ... import check_dense,random_array, random_ttarray, random_ttslice
from ... import DENSE_SHAPE
import pytest
CONTRACTS={
0:[((),())],
1:[((),()),((0,),(0,))],
2:[((),()),((0,),(1,)),((0,1),(-1,0)),(1,0)],
3:[]
}
def _calc_cluster(c1,c2,con):
    if isinstance(con,int):
        con0=tuple(range(len(c1[0])-con,len(c1[0])))
        con1=tuple(range(0,con))
        print(con0,con1)
    elif isinstance(con[0],int):
        con0=(con[0],)
        con1=(con[1],)
    else:
        con0=[c if c>=0 else len(c1[0])+c for c in con[0]]
        con1=[c if c>=0 else len(c2[0])+c for c in con[1]]
    cc=[tuple(c for i,c in enumerate(cc1) if i not in con0)+tuple(c for i,c in enumerate(cc2) if i not in con1) for cc1,cc2 in zip(c1,c2)]
    return tuple(cc)

def test_tensordot_ttarray_ttarray_dense(seed_rng):
    for shape,cls in DENSE_SHAPE.items():
        ar1=random_array(shape,float)
        ar2=random_array(shape[::-1],float)
        contracts=CONTRACTS[len(shape)]
        for con in contracts:
            ars=np.tensordot(ar1,ar2,con)
            c1=cls[0]
            c2=cls[1] if len(cls)>1 else cls[0]
            ttar1=tt.array(ar1,cluster=c1)
            ttar2=tt.array(ar2,cluster=c1)
            ttars1=tt.tensordot(ttar1,ttar2,con)
            check_dense(ttars1,ars,_calc_cluster(c1,c1,con),None,tt.TensorTrainArray)
        for n in range(1,len(shape)):
            ars=np.tensordot(ar1,ar2,n)
            c1=cls[0]
            c2=cls[1] if len(cls)>1 else cls[0]
            ttar1=tt.array(ar1,cluster=c1)
            ttar2=tt.array(ar2,cluster=c1)
            ttars1=tt.tensordot(ttar1,ttar2,n)
            check_dense(ttars1,ars,_calc_cluster(c1,c1,n),None,tt.TensorTrainArray)
