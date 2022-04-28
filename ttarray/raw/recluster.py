from .array_conversion import dense_to_ttslice,ttslice_to_dense, trivial_decomposition
import numpy as np
def recluster(ttslice,cluster_new,decomposition=trivial_decomposition):
    '''
        Straightforward reclustering algorithm: join sites until both clusterings
        and then use dense_to_ttslice to introduce new cluster
        TODO: overhaul this function, incredibly fragile right now
    '''
    ttiter=ttslice.__iter__()
    ret=[]
    ttcur=[]
    ttcl=(1,)*len(cluster_new[0])
    for cl in cluster_new:
        try:
            while any(x%y!=0 for x,y in zip(ttcl,cl)) or len(ttcur)==0:
                    ttcur.append(next(ttiter))
                    ttcl=tuple(x*y for x,y in zip(ttcur[-1].shape[1:-1],ttcl))
        except StopIteration:
            if len(ttcur)==0:
                inds=(slice(None),)+(None,)*len(cl)+(slice(None),)
                ret.append(np.eye(ret[-1].shape[-1],like=ret[-1])[inds])
                continue
            else:
                pass
        ttcl=tuple(x//y for x,y in zip(ttcl,cl))
        if not all(x==1 for x in ttcl):
            r,t=dense_to_ttslice(ttslice_to_dense(ttcur),(cl,ttcl),decomposition)
            ret.append(r)
            ttcur=[t]
        else:
            ret.append(ttslice_to_dense(ttcur))
            ttcur=[]
    for rest in ttiter:
        ret[-1]=ret[-1]@np.squeeze(rest,axis=tuple(range(1,len(rest.shape)-1)))
    return ret
