HANDLER_FUNCTION={}
HANDLER_UFUNC={}
def implement_function(func):
    def implement_function_decorator(f):
        HANDLER_FUNCTION[func]=wrap_function(f)
        return f
    return implement_function_decorator
def implement_ufunc(ufunc,method):
    def implement_ufunc_decorator(f):
        HANDLER_FUNCTION[(ufunc,method)]=wrap_ufunc(f)
        return f
    return implement_ufunc_decorator

def wrap_function(fun):
    def inner(func,types,args,kwargs):
        return fun(*args,**kwargs)
    return inner

def wrap_ufunc(fun):
    def inner(ufunc,method,args,kwargs):
        return fun(*args,**kwargs)
    return inner

def dispatch_array_ufunc(ufunc,method,args,kwargs):
    ret=HANDLER_UFUNC[(ufunc,method)](ufunc,method,args,kwargs)
    return ret

def dispatch_array_function(function,types,args,kwargs):
    ret=HANDLER_FUNCTION[function](function,types,args,kwargs)
    return ret
