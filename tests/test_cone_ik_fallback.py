"""The experimental fallback must preserve the original checks and order."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import phase2_controller as controller_module


@pytest.mark.parametrize('enabled,collision,expected', [(False, False, False), (True, False, True), (True, True, False)])
def test_cone_fallback_is_opt_in_and_checked(monkeypatch, enabled, collision, expected):
    common = ModuleType('common')
    ik = ModuleType('common.ik')
    robot_module = ModuleType('common.robot')
    robot_module.ARM_JOINTS, robot_module.SITE, robot_module.DOWN = [], 'site', np.array([0., 0., -1.])
    directions_seen, checked = [], []
    def solve(model, data, site, target, direction, joints):
        marker = 0 if direction is None else (1 if np.allclose(direction, robot_module.DOWN) else 2)
        directions_seen.append(marker)
        return np.array([marker,0.,0.,0.,0.]), 0.
    ik.solve_ik = solve
    monkeypatch.setitem(sys.modules, 'common', common)
    monkeypatch.setitem(sys.modules, 'common.ik', ik)
    monkeypatch.setitem(sys.modules, 'common.robot', robot_module)
    monkeypatch.setattr(controller_module.mujoco, 'MjData', lambda model: SimpleNamespace(qpos=np.zeros(5)))
    def check(robot, q, target, min_direction_dot):
        checked.append((q[0], min_direction_dot))
        return {'valid': q[0] == 2 and min_direction_dot == .70, 'position_error_m': .003, 'downward_axis_dot': .766}
    monkeypatch.setattr(controller_module, 'check_pose', check)
    monkeypatch.setattr(controller_module, 'inspect_path', lambda *a, **kw: {'first_penetration': {'sample': 1} if collision else None})
    robot = SimpleNamespace(q_arm=np.zeros(5), arm_qadr=np.arange(5),
        model=SimpleNamespace(key_qpos=np.zeros((1, 5))), data=SimpleNamespace(qpos=np.zeros(5), time=1.))
    controller = controller_module.RecoveryController(robot, experimental_cone_ik=enabled)
    moves = []
    controller.move = lambda *a, **kw: moves.append(a)
    controller.target([.05,.22,.12], 'place', -.1, allow_cube_contact=True)
    assert bool(moves) is expected
    assert 2 in directions_seen if enabled else 2 not in directions_seen
    if enabled:
        assert all(marker != 2 for marker in directions_seen[:24])
        assert (2, .70) in checked
        assert any(e['state'] == 'experimental_cone_ik' for e in controller.events) is expected
