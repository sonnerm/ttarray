from .. import raw

def canonicalize(ar,center=-1,oldcenter=None,svs=None,inplace=True):
    if oldcenter is None:
        return raw.canonicalize(ar,center)
    elif svs is None:
        return raw.shift_orthogonality_center(ar,oldcenter,center)
    else:
        return raw.shift_orthogonality_center_with_singular_values(ar,oldcenter,newcenter,svs)


def is_canonical(ar,center=None):
    if center is not None:
        return raw.is_canonical(ar,center)
def canonical_component(ar,i,center=None,svs=None):
    pass
def singular_values(ar,center=None):
    pass
