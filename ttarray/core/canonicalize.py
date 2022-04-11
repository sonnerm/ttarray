import numpy.linalg as la

def canonicalize(ar,center=None,copy=False,qr=la.qr):
    return ar.canonicalize(center,center,copy,qr)

def is_canonical(ar,center=None):
    return ar.is_canonical(center)

def singular_values(ar,center=None,copy=False,qr=la.qr,svd=la.svd):
    return ar.singular_values(center,copy,qr,svd)
