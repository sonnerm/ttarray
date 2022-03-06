import numpy as np
import functools
def trivial_decomposition(M):
    return np.eye(M.shape[0],like=M),M
def find_balanced_cluster(shape):
    if len(shape)==0:
        return ((),)
    cluster=[]
    for s in shape:
        clc=[]
        if s==1:
            cluster.append([1])
            continue
        while s%2==0:
            clc.append(2)
            s//=2
        while s>1:
            i=0
            for i in range(3,int(np.sqrt(s))+1,2):
                if s%i==0:
                    clc.append(i)
                    s//=i
                    i=0
                    break
            if i==0:
                clc.append(s)
                break
        cluster.append(clc[::-1])
    maxlen=max(len(c) for c in cluster)
    for c in cluster:
        if len(c)<maxlen:
            c.extend([1]*(maxlen-len(c)))
    return tuple(zip(*cluster))
def _product(seq):
    return functools.reduce(lambda x,y:x*y,seq,1)
def array_to_ttslice(a,cluster,decomposition):
    '''
        Converts an array to a ttslice with a given cluster using a given decomposition
    '''
    mps=[]
    print(a.shape)
    car=a.reshape(a.shape[0],-1)
    for ds in cluster:
        car=car.reshape((car.shape[0]*_product(ds),-1))
        q,r=decomposition(car)
        mps.append(q.reshape((-1,)+ds+(q.shape[1],)))
        car=r
    mps[-1]=np.tensordot(mps[-1],r,axes=((-1,),(0,)))
    return mps

def ttslice_to_array(ttslice):
    '''
        Converts ttslice as list of matrices into an array of the same kind as the constituent matrices
    '''
    ret=ttslice[0]
    r=len(ret.shape)-2
    trp=[0]*(2*r+2)
    trp[1:-1:2]=range(1,r+1)
    trp[2::2]=range(r+1,2*r+1)
    trp[-1]=2*r+1
    print(trp)
    for m in ttslice[1:]:
        ret=np.tensordot(ret,m,axes=((-1),(0)))
        ret=ret.transpose(trp)
        rshp=[ret.shape[0]]+[ret.shape[2*i+1]*ret.shape[2*i+2] for i in range(r)]+[ret.shape[-1]]
        ret=ret.reshape(rshp)
    return ret
