numpy and ttarray
==============
``numpy`` is the de-facto standard library for dealing with numerical arrays in
python. ``ttarray`` uses the  ``__array_function__`` and ``__array_ufunc__``
protocols to make TensorTrainArray_ and TensorTrainSlice_ 'feel' like regular
numpy.ndarray's. If functions has however the drawback that tensor train
specific parameters like bond-dimensions can not be provided this way. In the
future we plan on implementing some kind of global configuration to address this
shortcoming.
