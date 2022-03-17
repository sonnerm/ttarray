import numpy as np
def construt_diag_ttslice(ttslice,k=0):
    if k!=0:
        raise NotImplementedError("Not there yet")
    raise NotImplementedError("Not yet")
def extract_diag_ttslice(ttslice,offset=0,axis1=0,axis2=1):
    if offset!=0:
        raise NotImplementedError("Not implemented yet")
        #gonna get more complicated in that case
    return [np.diagonal(x,axis1=axis1+1,axis2=axis2+1) for x in ttslice]
