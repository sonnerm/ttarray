from .. import raw
import numpy.linalg as la

def canonicalize(ar,center=-1,oldcenter=None,svs=None,inplace=True,qr=la.qr):
    if not inplace:
        ar=ar.copy()
    if oldcenter is None:
        raw.canonicalize(ar.asmatrices_unchecked(),center)
    elif svs is None:
        raw.shift_orthogonality_center(ar.asmatrices_unchecked(),oldcenter,center)
    else:
        raw.shift_orthogonality_center_with_singular_values(ar.asmatrices_unchecked(),oldcenter,newcenter,svs)
    return ar

def is_canonical(ar,center=None):
    if center is not None:
        return raw.is_canonical(ar.asmatrices_unchecked(),center)
    else:
        return raw.find_orthogonality_center(ar.asmatrices_unchecked())
def singular_values(ar,center=None,inplace=False,qr=la.qr,svd=la.svd):
    if center is None:
        ar=canonicalize(ar,-1,inplace=inplace,qr=qr)
        center=-1
    return raw.singular_values(ar.asmatrices_unchecked(),center,svd=svd)
