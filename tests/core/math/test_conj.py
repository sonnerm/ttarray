import ttarray as tt
import numpy as np
from ...helper import random_array
import pytest
def test_array_conj_complex(seed_rng):
    ar=random_array((24,35),complex)
    arc=ar.conj()
    ttar1=tt.array(ar)
    ttar1c=ttar1.conj()
    assert arc==pytest.approx(ttar1c.todense())
    assert ar==pytest.approx(ttar1.todense())#didn't change ttar1
    ttar2c=tt.conj(ttar1)
    assert arc==pytest.approx(ttar2c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar3c=np.conj(ttar1)
    assert arc==pytest.approx(ttar3c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar4c=np.conjugate(ttar1)
    assert arc==pytest.approx(ttar4c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar5c=tt.conjugate(ttar1)
    assert arc==pytest.approx(ttar5c.todense())
    assert ar==pytest.approx(ttar1.todense())

def test_slice_conj_complex(seed_rng):
    ar=random_array((2,12,3),complex)
    arc=ar.conj()
    ttar1=tt.slice(ar)
    ttar1c=ttar1.conj()
    assert arc==pytest.approx(ttar1c.todense())
    assert ar==pytest.approx(ttar1.todense())#didn't change ttar1
    ttar2c=tt.conj(ttar1)
    assert arc==pytest.approx(ttar2c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar3c=np.conj(ttar1)
    assert arc==pytest.approx(ttar3c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar4c=np.conjugate(ttar1)
    assert arc==pytest.approx(ttar4c.todense())
    assert ar==pytest.approx(ttar1.todense())
    ttar5c=tt.conjugate(ttar1)
    assert arc==pytest.approx(ttar5c.todense())
    assert ar==pytest.approx(ttar1.todense())

def test_array_conj_real(seed_rng):
    ar=random_array((24,35),float)
    ttar1=tt.array(ar)
    ttar1c=ttar1.conj()
    ttar2c=tt.conj(ttar1)
    ttar3c=np.conj(ttar1)
    ttar4c=np.conjugate(ttar1)
    ttar5c=tt.conjugate(ttar1)
    assert ar==pytest.approx(ttar1.todense())#didn't change ttar1
    assert ar==pytest.approx(ttar1c.todense())#didn't change ttar1
    assert ar==pytest.approx(ttar2c.todense())#didn't change ttar1
    assert ar==pytest.approx(ttar3c.todense())#didn't change ttar1
    assert ar==pytest.approx(ttar4c.todense())#didn't change ttar1
    assert ar==pytest.approx(ttar5c.todense())#didn't change ttar1
