from .config import DEFAULT_OPTIONS
HANDLER_FUNCTION={}
HANDLER_UFUNC={}
def implement_function(algo,func):
    def implement_function_decorator(f):
        HANDLER_FUNCTION[algo][func]=f
        return f
    return implement_function_decorator
def implement_ufunc(algo,ufunc,method):
    def implement_ufunc_decorator(f):
        HANDLER_FUNCTION[algo][(ufunc,method)]=f
        return f
    return implement_ufunc_decorator

def _dispatch_array_ufunc(ufunc,method,args,kwargs):
    algo=kwargs.get(["mp_algo"],DEFAULT_OPTIONS["mp_algo"])
    return HANDLER_FUNCTION[algo][(ufunc,method)](ufunc,method,args,kwargs)

def _dispatch_array_function(function,types,args,kwargs):
    algo=kwargs.get(["mp_algo"],DEFAULT_OPTIONS["mp_algo"])
    return HANDLER_FUNCTION[algo][function](function,types,args,kwargs)
