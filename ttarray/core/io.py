from .array import TensorTrainArray
from .slice import TensorTrainSlice
import numpy as np
TENSOR_TRAIN_ARRAY_TYPE=1
TENSOR_TRAIN_SLICE_TYPE=2
def loadhdf5(hdf5obj,name=None):
    if name is not None:
        hdf5obj=hdf5obj[name]
    if "type" not in hdf5obj.keys():
        raise ValueError("'type' member not found in hdf5obj")
    if np.array(hdf5obj["type"]) == TENSOR_TRAIN_ARRAY_TYPE:
        cls=TensorTrainArray
    elif np.array(hdf5obj["type"]) == TENSOR_TRAIN_SLICE_TYPE:
        cls=TensorTrainSlice
    else:
        raise ValueError("'%s' is an unknown type"%hdf5obj["type"])
    Ws=[]
    for i in range(int(np.array(hdf5obj["L"]))):
        Ws.append(np.array(hdf5obj["M_%i"%i]))
    return cls.frommatrices(Ws)


def savehdf5(obj,hdf5obj,name=None):
    if name is not None:
        hdf5obj=hdf5obj.create_group(name)
    if isinstance(obj,TensorTrainArray):
        hdf5obj["type"]=TENSOR_TRAIN_ARRAY_TYPE
    elif isinstance(obj,TensorTrainSlice):
        hdf5obj["type"]=TENSOR_TRAIN_SLICE_TYPE
    else:
        return NotImplemented
    L=obj.L
    hdf5obj["L"]=L
    Ms=obj.tomatrices_unchecked()
    for i,m in enumerate(Ms):
        hdf5obj["M_%i"%i]=np.array(m)
