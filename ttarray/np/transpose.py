import numpy as np
from ..dispatch import implement_function,wrap_function
@implement_function
def transpose(a, axes=None):
    r=len(a.shape)
    if axes==None:
        axes=list(range(r))[::-1]
    if isinstance(a,TensorTrainArray):
        if axes[0]!=0 and axes[-1]!=r:
            return NotImplemented #doesn't work for the MatrixProductSlice
        else:
            return TensorTrainSlice.from_matrix_list([x.transpose(axes) for x in a.as_matrix_list()])
    if isinstance(a,TensorTrainArray):
        naxes=[0]+axes+[r]
        return TensorTrainSlice.from_matrix_list([x.transpose(naxes) for x in a.as_matrix_list()])
