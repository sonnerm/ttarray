import numpy as np
from .. import implement_function
@implement_function
def tensordot(a,b,axes=2):
    if isinstance(a,TensorTrainArray):
        if isinstance(b,TensorTrainArray):
            a=recluster(a,b.cluster)
            return raw.tensordot(a,b)
        else:#other array like
            pass
    else:
        pass
