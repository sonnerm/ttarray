from .dispatch import implement_function
@implement_function()
def reshape(ar,newshape,newcluster=None,copy=False,*,order=None):
    return ar.reshape(newshape,newcluster,copy)
def recluster(ar,newcluster=None,axes=None,copy=False):
    return ar.recluster(newcluster,axes=axes,copy=copy)
