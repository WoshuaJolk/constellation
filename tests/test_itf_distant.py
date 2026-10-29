import numpy as np
import pytest
from magictools.itf.distant import distant_seeds
from magictools.itf.kernels import MU


def test_bound_interval_accounts_for_observer_velocity_and_preserves_inputs():
    t=np.array([0.,1.,2.]);u=np.tile([1.,0.,0.],(3,1));o=np.array([[-.01,0,0],[0,0,0],[.01,0,0.]])
    copies=[x.copy() for x in [t,u,o]]
    seeds,receipt=distant_seeds(t,u,o,distances=[40.,1000.],radial_points=9)
    assert len(seeds)==18
    assert all(row['radial_center_au_day']==pytest.approx(-.01) for row in receipt['distances'])
    for state,epoch in seeds:
        assert epoch==1.
        assert .5*np.dot(state[3:],state[3:])-MU/np.linalg.norm(state[:3])<0
    for first,after in zip(copies,[t,u,o]):np.testing.assert_array_equal(first,after)


def test_impossible_chord_is_reported_not_accepted_or_silently_replaced():
    t=np.array([0.,1.]);u=np.tile([1.,0.,0.],(2,1));o=np.array([[0.,0,0],[0,1.,0]])
    seeds,receipt=distant_seeds(t,u,o,distances=[1000.])
    assert seeds==[]
    assert receipt['distances'][0]['status'].endswith('not_an_exclusion')
    with pytest.raises(ValueError,match='chronological'):
        distant_seeds(t[::-1],u,o)
