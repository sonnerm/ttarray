import numpy as np
from ..ttarray import implement_function
def _normalize_axes(axes):
    if axes==None:
        axes=list(range(r))[::-1]
    axes=[a if a>0 else L+a for a in axes]
    return axes
@implement_function()
def transpose(a, axes=None):
    r=len(a.shape)
    axes=_normalize_axes(axes)
    if isinstance(a,TensorTrainSlice):
        if axes[0]!=0 and axes[-1]!=r:
            #recluster and then transpose?
            pass
        else:
            return TensorTrainSlice.frommatrices([x.transpose(axes) for x in a.as_matrix_list()])
    if isinstance(a,TensorTrainArray):
        naxes=[0]+axes+[r]
        return TensorTrainArray.frommatrices([x.transpose(naxes) for x in a.as_matrix_list()])
