import numpy as np
def trivial_decomposition(M):
    return np.eye(M.shape[0],like=M),M
def find_balanced_cluster(shape):
    cluster=[]
    for s in shape:
        clc=[]
        while s%2==1:
            clc.append(2)
            s//=2
        while s>1:
            for i in range(3,np.sqrt(s),2):
                if s%i==0:
                    clc.append(i)
                    s//=i
                    i=0
                    break
            if i!=0:
                clc.append(s)
                break
        cluster.append(clc[::-1])
    maxlen=max(len(c) for c in cluster)
    for c in cluster:
        if len(c)<maxlen:
            c.extend([1]*(maxlen-len(c)))
    return list(zip(*cluster))


def array_to_ttslice(a,cluster,decomposition):
    '''
        Converts an array to a ttslice with a given cluster using a given decomposition
    '''
    mps=[]
    car=a.reshape(a.shape[0],-1)
    for ds in zip(cluster):
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
    for m in ttslice[1:]:
        ret=np.tensordot(ret,m,axes=((-1),(0)))
    return ret
