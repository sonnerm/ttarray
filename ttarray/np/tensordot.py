import numpy as np
from .. import implement_function
@implement_function("exact",np.tensordot)
@wrap_function
def tensordot(a,b,axes=2):
    if isinstance(a,MatrixProductBase):
        if isinstance(b,MatrixProductBase):
            if len(a.M)==len(b.M):
                pass
        else:#other array like
            pass
    else:
        pass
