Ownership of Data
===========================
Some operations like truncation require to bring tensor trains into canonical
form first. This step can be skipped if it is known that the tensor train is
already canonicalized. Since it is expensive to check if a given tensor train is
canonical, the TensorTrainSlice_ class stores the position of orthogonality
center if it is known from previous calculations that the tensor train is
canonical. This is only useful, if we can somehow ensure that the meta
information actually is true given the underlying data, which is often refered
to as the *invariant* of the data structure. Methods on TensorTrainSlice_ which
manipulate data can and do take care of keeping invariants automatically as
apropriate. Invariants could be violated, when the underlying data is modified
directly through outside references. For this reason, **array classes assume
that the data is exclusively owned by and can only be accessed through their
respective instance**. This could be enforced by never returning any direct
references to internal data through public methods and copying any external data
before it is stored, a concept known as 'encapsulation'. However, not allowing
any outside references comes at a steep cost. Firstly, copying of potentially
large amount of data is an expensive operation both in terms of memory and
computation time. Doing it when not necessary degrades performance
significantly. Secondly, not being able to directly manipulate the internal data
forces users who wish to implement their own algorithms to only use public
methods for manipulating data which might not be efficient. Thirdly, the
invariant preservation in public methods needs to be conservative. Since it is
expensive to determine if a tensor train is canonical, in many cases a public
method needs to assume that the tensor train is no longer canonical even though
in many particular cases it might still be.

For the reasons outlined above, we choose to have it both ways. Methods and
properties *except* those which end in ``_unchecked`` are guaranteed to never
leak any references, preserve all invariants and copy the data if necessary.
This avoids any data dependencies and outside references to internal data. If
invariants are violated in code that does not use any ``_unchecked`` methods
this is due to a bug in this library (and it should be reported by opening an
issue on github 😉).

Methods and properties which end in ``_unchecked`` on the other hand return
references and do not copy data. They also do nothing to preserve invariants. It
is the task of the consumer of this library to ensure that invariants are
preserved if using ``_unchecked`` methods or properties.
