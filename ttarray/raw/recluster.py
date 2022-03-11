from .array_conversion import dense_to_ttslice,ttslice_to_dense
def recluster(ttslice,cluster_new,decomposition):
    '''
        Straightforward reclustering algorithm: join sites until both clusterings
        and then use dense_to_ttslice to introduce new cluster
    '''
    ttiter=ttslice.__iter__()
    cliter=cluster_new.__iter()
    ret=[]
    ttcur=[]
    ttcl=(1,)*len(cluster_new[0])
    clcl=(1,)*len(cluster_new[0])
    try:
        while 1:
            if any(x==1 and y!=1 for x,y in zip(clcl,ttcl)):
                clcl=tuple(x*y for x,y in zip(next(cliter),clcl))
            elif all(x%y==0 for x,y in zip(ttcl,clcl)):
                ttcl=tuple(x//y for x,y in zip(ttcl,clcl))
                r,t=dense_to_ttslice(ttslice_to_dense(ttcur),(clcl,ttcl),decomposition=decomposition)
                ret.append(r)
                ttcur=[t]
            else:
                ttcur.append(next(ttiter))
                ttcl=tuple(x*y for x,y in zip(ttcur[-1].shape[1:-1],ttcl))
    except StopIteration:
        return ret
