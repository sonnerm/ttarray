from .array_conversion import dense_to_ttslice,ttslice_to_dense
def recluster(ttslice,cluster_new,decomposition):
    '''
        Straightforward reclustering algorithm: join sites until both clusterings
        and then use dense_to_ttslice to introduce new cluster
    '''
    ttiter=ttslice.__iter__()
    ret=[]
    ttcur=[]
    ttcl=(1,)*len(cluster_new[0])
    for cl in cluster_new:
        if all(x%y==0 for x,y in zip(ttcl,cl)):
            ttcl=tuple(x//y for x,y in zip(ttcl,cl))
            r,t=dense_to_ttslice(ttslice_to_dense(ttcur),ttcl,decomposition)
            ret.append(r)
            ttcur.append(t)
        else:
            ttcur.append(next(ttiter))
            ttcl=tuple(x*y for x,y in zip(ttcur[-1].shape[1:-1],ttcl))
    return ret
