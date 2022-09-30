from .. import implement_function
@implement_function()
def transpose(a, axes=None):
    a.transpose(axes)
