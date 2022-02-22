from collections import defaultdict
from .dense import dense_array_ufunc,dense_array_function
HANDLER_FUNCTION["exact"]=defaultdict(lambda: dense_function)
HANDLER_UFUNC["exact"]=defaultdict(lambda: dense_ufunc)
