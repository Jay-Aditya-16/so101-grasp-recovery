import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_transfer_ik import cone_directions


def test_direction_cone_has_correct_tilt_and_unit_length():
    dirs = np.array(list(cone_directions(40, 8)))
    assert dirs.shape == (8, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1)
    assert np.allclose(-dirs[:, 2], np.cos(np.deg2rad(40)))
    assert np.allclose(dirs[:, :2].sum(axis=0), 0, atol=1e-15)


@pytest.mark.parametrize('tilt,count', [(-1,8),(90,8),(30,0)])
def test_invalid_direction_cone(tilt, count):
    with pytest.raises(ValueError):
        list(cone_directions(tilt, count))


def test_rigid_attachment_composes_rotation_and_translation():
    from diagnose_transfer_ik import attached_pose
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    p, r = attached_pose(np.array([1., 2., 3.]), rot, np.array([.1, 0, 0]), np.eye(3))
    assert np.allclose(p, [1., 2.1, 3.])
    assert np.allclose(r, rot)
