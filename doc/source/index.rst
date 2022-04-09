.. ttarray documentation master file, created by
   sphinx-quickstart on Wed Apr  6 14:09:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ttarray's documentation!
===================================
``ttarray`` provides a library for calculations with multi dimensional arrays
in the tensor train format. Vectors in this format are well known in the
computational quantum physics world as matrix product states (MPS). This format
allows us to represent and manipulate low-entangled quantum states in very
high-dimensional Hilbert spaces (even something like 2^200 dimensional !). It
turns out that many states of interests like ground states of local one
dimensional Hamiltonians or thermal density matrices have low entanglement and
can thus be efficiently worked with in tensor train form.

The package is build around the TensorTrainArray class which provides a numpy
compatible api and are designed to be used just as numpy arrays. Leveraging the
:py:func:`__array_function__` and :py:func:`__array_ufunc__` protocols this even
works for routines in the normal numpy namespace or in some third party
libraries. However, it is important to note that not all operations can be
performed efficiently on tensor trains! Additional methods specific to the
TensorTrain format are provided as well.

Under the hood, the data is stored as a normal python list of numpy (or
numpy-compatible) ndarrays which can be retrieved and manipulated manually if
necessary. The raw namespace provides the basic algorithm which can be aplied to
this format.

.. note::
  This project is at present very much a work in progress and should be
  considered experimental. Any feedback is welcome !

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ownership
   tensortrain

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
