import pytest
import ttarray as tt
from ..helper import random_ttarray,random_ttslice
def test_save_array(tmpdir,seed_rng):
    import h5py
    f=h5py.File(tmpdir.join("test_save_array.h5"),"w")
    tta=random_ttarray()
    ttb=tta.copy()
    tt.savehdf5(ttb,f,"nami")
    f.close()
    del ttb
    del f
    f2=h5py.File(tmpdir.join("test_save_array.h5"))
    ttc=tt.loadhdf5(f2["nami"])
    for a,b in zip(tta.M,ttc.M):
        assert a==pytest.approx(b)

def test_save_array_dict(seed_rng):
    tta=random_ttarray()
    ttb=tta.copy()
    sd={}
    tt.savehdf5(ttb,sd)
    del ttb
    ttc=tt.loadhdf5(sd)
    for a,b in zip(tta.M,ttc.M):
        assert a==pytest.approx(b)
def test_save_slice(tmpdir,seed_rng):
    import h5py
    f=h5py.File(tmpdir.join("test_save_slice.h5"),"w")
    tta=random_ttslice()
    ttb=tta.copy()
    tt.savehdf5(ttb,f)
    f.close()
    del ttb
    del f
    f2=h5py.File(tmpdir.join("test_save_slice.h5"))
    ttc=tt.loadhdf5(f2)
    for a,b in zip(tta.M,ttc.M):
        assert a==pytest.approx(b)

def test_save_slice_dict(seed_rng):
    tta=random_ttslice()
    ttb=tta.copy()
    sd={}
    tt.savehdf5(ttb,sd)
    del ttb
    ttc=tt.loadhdf5(sd)
    for a,b in zip(tta.M,ttc.M):
        assert a==pytest.approx(b)
