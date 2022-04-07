This package provides a library with numpy-compatible api for calculations with
multi dimensional arrays in the tensor train format. This format allows for the
(lossy) compressed storage of very high dimensional arrays (think 2^200 or
more). This is possible due to *truncation* of small singular values.

Vectors in the tensor train format are well known in the computational quantum
physics world as matrix product states (MPS). This format allows us to represent
and manipulate low-entangled quantum states in very high-dimensional Hilbert
spaces. It turns out that many states of interests like ground states of local
one dimensional Hamiltonians or thermal density matrices have low entanglement
and can thus be efficiently worked with in tensor train form. This approach has
opened the door to much progress in the numerical study of quantum systems.

The package is build around the TensorTrainArray class which provides a numpy
compatible api and are designed to be used just as numpy arrays. Leveraging the
``__array_function__`` and ``__array_ufunc__`` protocols this even works for
routines in the normal numpy namespace or in some third party libraries.
However, it is important to note that not all operations can be performed
efficiently on tensor trains! Additional methods specific to the TensorTrain
format are provided as well.

Under the hood, the data is stored as a normal python list of numpy (or
numpy-compatible) ndarrays which can be retrieved and manipulated manually. The
raw namespace provides the basic algorithm which can be aplied to this format.

.. note::
  At present this package is a work in progress and is not yet complete
