import numpy as np
import functools
def trivial_decomposition(M):
    if M.shape[0]<=M.shape[1]:
        return np.eye(M.shape[0],dtype=M.dtype,like=M),M
    else:
        return M,np.eye(M.shape[1],dtype=M.dtype,like=M)
def find_balanced_cluster(shape,L=None):
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
    if L is None:
        L=max(len(c) for c in cluster)
    for i,c in enumerate(cluster):
        if len(c)<L:
            c.extend([1]*(L-len(c)))
        elif len(c)>L:
            cl=c[:L]
            for ii,cc in enumerate(c[L:]):
                cl[ii%L]*=cc
            cluster[i]=cl
    return tuple(zip(*cluster))
def _product(seq):
    return functools.reduce(lambda x,y:x*y,seq,1) # python unbounded ints ftw!

def dense_to_ttslice(a,cluster,decomposition=trivial_decomposition):
    '''
        Converts an array to a ttslice with a given cluster using a given decomposition
    '''
    mps=[]
    enddim=a.shape[-1]
    cshape=a.shape[1:-1]
    tpose=[0]+list(range(1,2*len(cshape)+1,2))+list(range(2,2*len(cshape)+2,2))+[2*len(cshape)+1]
    car=a.reshape(a.shape[0],-1)
    for ds in cluster:
        nshape=(car.shape[0],)+sum(((d,cs//d) for d,cs in zip(ds,cshape)),())+(enddim,)
        car=np.reshape(car,nshape)
        car=np.transpose(car,tpose)
        cshape=tuple((cs//d) for d,cs in zip(ds,cshape))
        car=car.reshape(car.shape[0]*_product(ds),-1)
        q,r=decomposition(car)
        mps.append(q.reshape((-1,)+ds+(q.shape[1],)))
        car=r
    mps[-1]=np.tensordot(mps[-1],r,axes=((-1,),(0,)))
    return mps

def ttslice_to_dense(ttslice):
    '''
        Converts ttslice as list of matrices into an array of the same kind as the constituent matrices
    '''
    ret=ttslice[0]
    r=len(ret.shape)-2
    trp=[0]*(2*r+2)
    trp[1:-1:2]=range(1,r+1)
    trp[2::2]=range(r+1,2*r+1)
    trp[-1]=2*r+1
    for m in ttslice[1:]:
        ret=np.tensordot(ret,m,axes=((-1),(0)))
        ret=ret.transpose(trp)
        rshp=[ret.shape[0]]+[ret.shape[2*i+1]*ret.shape[2*i+2] for i in range(r)]+[ret.shape[-1]]
        ret=ret.reshape(rshp)
    return ret
def locate_tensor(i,cluster):
    pass
