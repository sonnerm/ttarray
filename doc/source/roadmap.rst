Roadmap
========
Immediate Goals:
----------------
* Provide a *mostly* complete replication of the ``numpy`` API.
* Use object-local configurations to determine behavior
* Implement a larger set of algorithms like DMRG, ttcross, ...

Far Goals:
------------
* Implement other TensorTrain/TensorNetwork formats like periodic tensor trains,
  tree tensor networks, MERA, PEPS, ...

Non-goals
------------
* Routines to create matrix product operators or matrix product states relating to specific physical models.
  It is hard to define a good general API this task since there are often
  multiple different conventions in use (just start with Pauli-matrices vs spin
  matrices or fermion quantization axis along Z or X). For this reason I believe
  this is better handled in a separate package. Check out for example freeferm
  or imcode.
* Conserved quantities
  This is probably better handled in a separate package which provides ndarray's
  with charge information attached. See the np_conserved subpackage of tenpy.
