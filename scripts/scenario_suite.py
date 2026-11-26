"""Controlled out-of-distribution scenes for the pinned SO-101 tutorial simulator."""
from contextlib import contextmanager

CASES = ("cube", "small_cube", "cuboid", "sphere", "green_cube", "two_objects", "drawer")

DRAWER = '''
    <body name="test_drawer" pos="0.25 -0.04 0.025">
      <joint name="drawer_slide" type="slide" axis="-1 0 0" range="0 0.10" damping="1.5" frictionloss="0.05"/>
      <geom name="drawer_base" type="box" size="0.045 0.045 0.005" mass="0.06" rgba="0.55 0.55 0.6 1"/>
      <geom name="drawer_front" type="box" pos="-0.045 0 0.02" size="0.004 0.045 0.025" mass="0.02" rgba="0.5 0.5 0.55 1"/>
      <geom name="drawer_handle" type="capsule" fromto="-0.058 -0.025 0.03 -0.058 0.025 0.03" size="0.007" mass="0.01" rgba="0.8 0.1 0.1 1"/>
    </body>
'''


@contextmanager
def scenario_scene(name):
    """Temporarily change the scene template; do not modify the tutorial checkout."""
    if name not in CASES:
        raise ValueError(f"Unknown scenario: {name}")
    from common import scene
    original = scene.SCENE_TEMPLATE
    template = original
    kwargs = {}
    cube_size = 'type="box" size="0.015 0.015 0.015"'
    if name == "small_cube":
        template = template.replace(cube_size, 'type="box" size="0.010 0.010 0.010"')
        template = template.replace('{cube_y} 0.015', '{cube_y} 0.010')
    elif name == "cuboid":
        template = template.replace(cube_size, 'type="box" size="0.025 0.012 0.015"')
    elif name == "sphere":
        template = template.replace(cube_size, 'type="sphere" size="0.018"')
        template = template.replace('{cube_y} 0.015', '{cube_y} 0.018')
    elif name == "green_cube":
        template = template.replace('rgba="0.9 0.1 0.1 1"', 'rgba="0.1 0.7 0.2 1"')
    elif name == "two_objects":
        template = template.replace(cube_size, 'type="box" size="0.025 0.012 0.015"')
        kwargs["extra_cubes"] = [("green", (0.24, 0.09))]
    elif name == "drawer":
        template = template.replace('{extra_cubes}', '{extra_cubes}' + DRAWER)
        template = template.replace('rgba="0.9 0.1 0.1 1"', 'rgba="0.9 0.1 0.1 0"')
        template = template.replace('{extra_qpos}" ctrl=', '{extra_qpos} 0" ctrl=')
    try:
        scene.SCENE_TEMPLATE = template
        yield kwargs
    finally:
        scene.SCENE_TEMPLATE = original
