"""Experimental observation-grounded recovery for the tutorial simulator only."""
import numpy as np
import mujoco

from object_geometry import estimate_object
from phase2_planner import check_pose, inspect_path, cone_directions


class RecoveryController:
    """One recovery attempt after a suspected miss; no simulator object state input."""

    def __init__(self, robot, fps=20, depth_gate=None, depth_view="top", retry_only=False, bandit=None, bandit_mode=None, experimental_cone_ik=False, grasp_candidate="center"):
        self.robot, self.fps = robot, fps
        self.state = "idle"
        self.reason = None
        self.events = []
        self.command = None
        self.depth_gate = depth_gate
        self.depth_view = depth_view
        self.retry_only = retry_only
        self.depth_checks = []
        self.bandit, self.bandit_mode = bandit, bandit_mode
        self.bandit_selection = None
        self.experimental_cone_ik = experimental_cone_ik
        self.grasp_heights = {"low": 0.015, "center": 0.020, "high": 0.025}
        if grasp_candidate not in self.grasp_heights:
            raise ValueError("unknown cube grasp candidate")
        self.grasp_candidate = grasp_candidate

    def log(self, state, reason=None):
        self.state, self.reason = state, reason
        self.events.append({"state": state, "time_s": float(self.robot.data.time), "reason": reason})
        print(f"Recovery: {state}" + (f" — {reason}" if reason else ""), flush=True)

    def move(self, target, duration, next_state, grip=None, allow_cube_contact=False):
        start = self.robot.q_arm.copy()
        path = inspect_path(self.robot, start, target, allow_cube_contact=allow_cube_contact)
        if path["first_penetration"] is not None:
            self.log("aborted", f"sampled path collision: {path['first_penetration']}")
            return False
        self.start_q, self.target_q = start, np.asarray(target).copy()
        self.steps, self.index, self.next_state = max(1, round(duration * self.fps)), 0, next_state
        self.grip = float(self.robot.data.ctrl[self.robot.grip_act]) if grip is None else grip
        self.log("moving", next_state)
        return True

    def target(self, xyz, next_state, grip, allow_cube_contact=False):
        from common.ik import solve_ik
        from common.robot import ARM_JOINTS, SITE, DOWN
        target = np.asarray(xyz, dtype=float)
        targets = [target]
        if next_state == "place":
            # Select among points inside the known box, keeping 15 mm margin
            # against the 30 mm scoring region; no cube-contact state is used.
            from placement_bandit import OFFSETS
            for dx, dy in OFFSETS[1:]:
                targets.append(target + np.array([dx, dy, 0.0]))
        action_ids = None
        scores = None
        if next_state == "place" and self.bandit is not None:
            from placement_bandit import ACTIONS
            try:
                ranking, scores = self.bandit.ranked(self.xy)
            except ValueError as exc:
                self.log("aborted", f"bandit context rejected: {exc}")
                return
            action_ids = list(ACTIONS)
            if self.bandit_mode == "active":
                targets = [targets[ACTIONS.index(action)] for action in ranking]
                action_ids = ranking
        seeds = [self.robot.q_arm, self.robot.model.key_qpos[0, self.robot.arm_qadr].copy()]
        checks = []
        # During transport, try position-only IK if the five-DOF arm cannot reach
        # the exact downward orientation, and independently limit tool tilt.
        directions = (DOWN, None) if allow_cube_contact else (DOWN,)
        # Keep the original search first. Only the explicit simulator experiment
        # samples tool axes inside the unchanged transfer orientation limit.
        passes = [("baseline", directions)]
        if self.experimental_cone_ik and next_state in ("place", "release") and allow_cube_contact:
            passes.append(("cone40", tuple(cone_directions())))
        for pass_name, pass_directions in passes:
            for candidate_index, candidate in enumerate(targets):
                for direction in pass_directions:
                    for seed in seeds:
                        scratch = mujoco.MjData(self.robot.model)
                        scratch.qpos[:] = self.robot.data.qpos
                        scratch.qpos[self.robot.arm_qadr] = seed
                        q, _ = solve_ik(self.robot.model, scratch, SITE, candidate, direction, ARM_JOINTS)
                        min_dot = 0.70 if next_state in ("place", "release") and (direction is None or pass_name == "cone40") else (0.90 if direction is None else 0.98)
                        check = check_pose(self.robot, q, candidate, min_direction_dot=min_dot)
                        checks.append(check)
                        if check["valid"]:
                            path = inspect_path(self.robot, self.robot.q_arm, q, allow_cube_contact=allow_cube_contact)
                            if path["first_penetration"] is None:
                                if pass_name == "cone40":
                                    self.events.append({"state": "experimental_cone_ik", "time_s": float(self.robot.data.time),
                                                        "target": next_state, "direction": direction.tolist(),
                                                        "position_error_m": check["position_error_m"],
                                                        "downward_axis_dot": check["downward_axis_dot"]})
                                if next_state == "close":
                                    self.events.append({"state": "grasp_candidate", "time_s": float(self.robot.data.time),
                                                        "candidate": self.grasp_candidate, "target_xyz_m": candidate.tolist(),
                                                        "predicted_site_xyz_m": check["site_xyz_m"],
                                                        "position_error_m": check["position_error_m"],
                                                        "downward_axis_dot": check["downward_axis_dot"]})
                                if next_state == "place":
                                    self.place_xy = candidate[:2].copy()
                                    if action_ids is not None:
                                        self.bandit_selection = {"action": action_ids[candidate_index],
                                            "observed_xy_m": self.xy.tolist(), "mode": self.bandit_mode,
                                            "scores": scores}
                                        self.events.append({"state": "bandit_selection", "time_s": float(self.robot.data.time),
                                                            **self.bandit_selection})
                                    self.events.append({"state": "box_target", "time_s": float(self.robot.data.time),
                                                        "xy_m": self.place_xy.tolist()})
                                self.move(q, 1.5, next_state, grip, allow_cube_contact)
                                return
        self.log("aborted", f"no IK solution with validated pose and sampled path: {checks}")

    def start(self):
        home = self.robot.model.key_qpos[0, self.robot.arm_qadr].copy()
        self.move(home, 4.0, "observe")

    def check_gripper(self, stage):
        """A closed jaw blocked by the known cube is evidence, not proof, of a grasp."""
        measured = float(self.robot.gripper_opening())
        plausible = bool(np.isfinite(measured) and 0.08 <= measured <= 0.8)
        self.events.append({"state": "gripper_check", "time_s": float(self.robot.data.time),
                            "reason": stage, "measured_rad": measured, "plausible": plausible})
        print(f"Gripper check ({stage}): {measured:.3f} rad; plausible={plausible}", flush=True)
        if not plausible:
            self.log("aborted", f"no plausible gripper obstruction at {stage}")
        return plausible

    def step(self, top_rgb, wrist_rgb=None):
        robot = self.robot
        if self.state == "moving":
            self.index += 1
            a = min(1.0, self.index / self.steps)
            command = np.r_[(1 - a) * self.start_q + a * self.target_q, self.grip]
            if self.index == self.steps:
                self.log(self.next_state)
            return command
        if self.state == "observe":
            if self.retry_only:
                self.log("policy_retry", "restart ACT from the observation pose")
                return robot.data.ctrl[:6].copy()
            cam = robot.model.camera("top").id
            obj = estimate_object(top_rgb, robot.data.cam_xpos[cam], robot.data.cam_xmat[cam], robot.model.cam_fovy[cam])
            if obj is None:
                self.log("aborted", "object is not reliably visible")
            else:
                if self.depth_gate is not None and self.depth_view == "top":
                    decision = self.depth_gate(top_rgb)
                    self.depth_checks.append({"time_s": float(robot.data.time), **decision})
                    print(f"Depth gate: {decision}", flush=True)
                    if not decision["accepted"]:
                        self.log("aborted", f"depth gate: {decision['reason']}")
                        return robot.data.ctrl[:6].copy()
                self.xy = np.asarray(obj["world_xy_m"])
                self.log("open", f"observed cube at {self.xy.tolist()}")
                self.wait = round(0.7 * self.fps)
        elif self.state == "open":
            self.wait -= 1
            if self.wait <= 0:
                self.target([*self.xy, 0.08], "descend", 1.2)
            return np.r_[robot.q_arm, 1.2]
        elif self.state == "descend":
            if self.depth_gate is not None and self.depth_view == "wrist":
                if wrist_rgb is None:
                    self.log("aborted", "wrist image unavailable")
                    return robot.data.ctrl[:6].copy()
                cam = robot.model.camera("wrist_cam").id
                wrist_obj = estimate_object(wrist_rgb, robot.data.cam_xpos[cam], robot.data.cam_xmat[cam], robot.model.cam_fovy[cam])
                if wrist_obj is None:
                    self.log("aborted", "object not reliably visible from wrist")
                    return robot.data.ctrl[:6].copy()
                wrist_xy = np.asarray(wrist_obj["world_xy_m"])
                disagreement = float(np.linalg.norm(wrist_xy - self.xy))
                decision = self.depth_gate(wrist_rgb)
                self.depth_checks.append({"time_s": float(robot.data.time), "view": "wrist", "wrist_xy_m": wrist_xy.tolist(),
                                          "overhead_xy_m": self.xy.tolist(), "view_disagreement_m": disagreement, **decision})
                print(f"Wrist depth gate: {self.depth_checks[-1]}", flush=True)
                if not decision["accepted"] or disagreement > 0.01:
                    self.log("aborted", "wrist depth or camera agreement check failed")
                    return robot.data.ctrl[:6].copy()
                self.xy = wrist_xy
            self.target([*self.xy, self.grasp_heights[self.grasp_candidate]], "close", 1.2, allow_cube_contact=True)
        elif self.state == "close":
            self.log("closing")
            self.wait = round(1.0 * self.fps)
        elif self.state == "closing":
            self.wait -= 1
            if self.wait <= 0:
                if self.check_gripper("after_closure"):
                    self.target([*self.xy, 0.12], "transport", -0.1, allow_cube_contact=True)
            return np.r_[robot.q_arm, -0.1]
        elif self.state == "transport":
            # The box pose is fixed task configuration in this tutorial scene.
            if self.check_gripper("after_lift"):
                self.target([0.05, 0.22, 0.12], "place", -0.1, allow_cube_contact=True)
        elif self.state == "place":
            if self.check_gripper("after_transport"):
                self.target([*self.place_xy, 0.06], "release", -0.1, allow_cube_contact=True)
        elif self.state == "release":
            self.log("releasing")
            self.wait = round(1.0 * self.fps)
        elif self.state == "releasing":
            self.wait -= 1
            if self.wait <= 0:
                self.log("done")
            return np.r_[robot.q_arm, 1.2]
        elif self.state == "aborted":
            return np.r_[robot.q_arm, float(robot.data.ctrl[robot.grip_act])]
        return robot.data.ctrl[:6].copy()
