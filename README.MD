This package provides a library for computations with matrix product arrays
(mpa), a generalization of matrix product states and matrix product operators
which are used extensively in computational quantum physics. These objects are
known as tensor trains in the mathematical community.

The different methods "mp_algo" have different guarantees:
  - "dense" converts each mpa into a dense numpy array. This is mostly useful
    for testing other implementations and as a fallback in case other
    implementations can not handle a specific operation (yet). Since mpas can
    represent very high dimensional arrays, this method might fail allocating
    and result in a MemoryError. To prevent this the parameter "dense_maxdim"
    should be set to a reasonable size.
  - "exact" leaves the matrix product form intact but performs every computation
    exactly. Currently this is the default. "exact_maxdim"
  - "svd" the result is equivalent to apply a truncation of the singular values after
    applying the operation exactly. Depending on the operation this might use a faster
    algorithm to reach the same result.
  - "var" variational method optimizes in some cases, this algorithm might get
    stuck in local minima and not reach the optimal "svd" solution
If other packages for matrix product states are installed, it is possible to use
them instead of our implementations:
  - "tenpy" uses the tenpy library
  - "quimb" uses the quimb library
