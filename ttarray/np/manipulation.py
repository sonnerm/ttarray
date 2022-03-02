from ..dispatch import implement_function
@implement_function(np.shape)
def shape(ar):
    return ar.shape
