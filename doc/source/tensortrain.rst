Tensor Train format
===================
Tensor Trains are a way to store, manipulate and compress very high-dimensional
arrays. Array elements of an array :math:`A^{\{\sigma_k\}_k}` in tensor train format are defined by
a set of :math:`L` matrices :math:`M^{(i),\{\sigma_{k, i}\}_k}_{\alpha_{i-1},\alpha_i}` through

.. math::

  A^{\{\{\sigma_{i,k}\}_i\}_k} = \sum_{\{\alpha_j\}_j} \prod_{i=1}^L M^{(i),\{\sigma_{k, i}\}_k}_{\alpha_{i-1},\alpha_i}

The contracted indices :math:`\alpha_i` are called internal or virtual indices,
the non-contracted indices :math:`\sigma_{k,i}` are called external or physical
indices. The dimensions of the array :math:`A` are the given by the *product* of
the dimension of the external indices of the underlying matrices :math:`M`. This
is the reason why tensor trains can represent exponentially high dimensional
arrays using only linear amounts of memory.
