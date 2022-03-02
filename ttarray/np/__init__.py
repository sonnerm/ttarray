from collections import defaultdict
import .config
import warnings

def default_array_function(func,types,args,kwargs):
    kwargs={**DEFAULT_OPTIONS,**kwargs,"mp_algo":"dense"}
    if kwargs["dense_fallback"]:
        return dispatch_array_function(func,types,args,kwargs)
    return NotImplemented

def default_array_ufunc(ufunc,method,args,kwargs):
    kwargs={**kwargs,"mp_algo":"dense"}
    if kwargs["dense_fallback"]:
        return dispatch_array_ufunc(ufunc,method,args,kwargs)
    return NotImplemented

HANDLER_FUNCTION["exact"]=defaultdict(lambda: default_array_function)
HANDLER_UFUNC["exact"]=defaultdict(lambda: default_array_ufunc)
HANDLER_UFUNC["exact"][NotImplemented]=default_array_ufunc
HANDLER_UFUNC["exact"][NotImplemented]=default_array_function #for now ...
from .tensordot import *
from .transpose import *
