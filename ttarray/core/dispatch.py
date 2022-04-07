HANDLER_FUNCTION_ARRAY={}
HANDLER_UFUNC_ARRAY={}
HANDLER_FUNCTION_SLICE={}
HANDLER_UFUNC_SLICE={}
def implement_function(func=None,selector=None):
    def implement_function_decorator(f):
        if func is None:
            fname=f.__name__
        else:
            fname=func
        if selector=="array" or selector is None:
            HANDLER_FUNCTION_ARRAY[fname]=f
        if selector=="slice" or selector is None:
            HANDLER_FUNCTION_SLICE[fname]=f
        return f
    return implement_function_decorator
def implement_ufunc(ufunc,method,selector=None):
    def implement_ufunc_decorator(f):
        if selector=="array" or selector is None:
            HANDLER_UFUNC_ARRAY[(ufunc,method)]=f
        if selector=="slice" or selector is None:
            HANDLER_UFUNC_SLICE[(ufunc,method)]=f
        return f
    return implement_ufunc_decorator
