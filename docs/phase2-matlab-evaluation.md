# Phase 2 matched evaluation and MATLAB analysis

This is a simulation research prototype for the fixed SO-101 tutorial scene and 30 mm red cube. The ACT checkpoint remains frozen. The recovery layer is hand-engineered and does not learn online. The two unknown-alert fallback cues are experimental and enabled explicitly with `--recover-unknown-empty --recover-unknown-stationary`.

## What changed

- For an `unknown` monitor event, three consecutive samples of a commanded-closed and measured-nearly-closed gripper can indicate an empty closure. Alternatively, after a half-second delay, three consecutive fresh images of the red region near its pre-closure position can indicate a stationary cube. Neither cue proves failure; they are tuned for this simulated cube and can misclassify unfamiliar objects, partial occlusion, or stale positions. `unknown` otherwise continues ACT.
- The recovery controller checks measured gripper opening after closing, lifting, and transport. The 0.08–0.8 rad window is an exploratory obstruction heuristic, not proof the object is held or that it has not slipped. It aborts if the check fails.
- The fixed box centre can be unreachable with the chosen IK tilt limit. The planner tries the centre followed by candidate points 15 mm within the known box region; it checks each pose's forward-kinematic error, joint limits, and sampled robot path. The selected point is used for the descent as well. The 1021 case was used to develop this feature.
- `--retry-only` provides a comparison that returns to the same observation pose and resets ACT, with no object-aware scripted grasp. Both recovery and retry get at most one intervention. The fixed 25 s simulated horizon includes reobservation and retry.

## Matched trials

| Seed | ACT alone | Return and retry ACT | Object-aware recovery |
|---|---|---|---|
| 1020 | No lift, no placement | No lift, no placement | 96.3 mm lift, placed |
| 1022 | 6.8 mm lift, no placement | 5.9 mm lift, no placement | 97.6 mm lift, placed |

All six trials were run live with the same pose per seed, checkpoint and 25 s simulated horizon. The 1020 recovery and retry ran before the stationary-cube fallback and box-candidate refinement; its monitor returned `suspected_miss`, so the fallback was not needed. The 1022 runs used the final fallback and box-candidate logic. Seed 1022 was first run after developing the logic on 1021; no formal preregistered test set exists. These are two paired examples, not a reliable population success rate or evidence of generalization across objects, lighting, cameras, or physical hardware.

The 1021 baseline and original fallback failed. After inspecting that trial and adding the stationary-cube cue, it lifted but aborted on IK tilt. A feasible interior box target then gave placement; 1021 is therefore a development case. The fallback was also tested on previously successful ACT seed 1003; ACT placed with no recovery intervention. The earlier 1011 empty-closure development case recovered to placement with a 25 s horizon. The 21 s run ended before the released cube had settled and must be scored as failure for that horizon.

The final code was additionally tested on seed 1022 with wrist Depth Anything V2 enabled: it placed, as did the same recovery without depth. At the wrist decision frame, the wrist estimate was 7.32 mm from the simulator cube position and the overhead estimate was 0.38 mm away. The model is a relative-depth ordering check, not metric depth; no accuracy benefit is established.

## MATLAB analysis

MATLAB R2025b analyzed the six matched CSV traces and generated `reports/phase2-matlab-matched.png` and `reports/phase2-matlab-matched.csv`. The cube height in those plots is simulator ground truth used only for evaluation. MATLAB measured peak lifts of 0 and 6.84 mm for ACT; 0 and 5.87 mm for retry; and 96.28 and 97.61 mm for recovery. In the recovery traces it measured joint openings of approximately 0.150–0.153 rad after closure and 0.147–0.148 rad after lift. These observations support testing the obstruction heuristic, but five or six traces cannot calibrate false positives or false negatives.

To reproduce the MATLAB analysis, use `scripts/export_rollout_csv.py --output-dir outputs/matlab-matched-input` followed by the six archives in `datasets/local-rollouts/phase2-{baseline,retry,recovery}-{1020,1022}.npz`; then run MATLAB `addpath('matlab'); analyze_phase2_batch('outputs/matlab-matched-input','outputs/matlab-matched')`. The JSON result for each trial and corresponding NPZ archive are stored in this private repository. Inputs, simulator-derived labels and actions should remain distinguished when building a training dataset.

## Limits before physical deployment or broader claims

The controller does not detect all failure modes, cannot prove a grasp from gripper angle, and its visual cue can be fooled. Sampled kinematic collision checks permit expected cube contact and do not check the swept held object, continuous clearance, forces, or real-world calibration. The MATLAB plots corroborate simulator traces; they are not an independent physics validation or physical robot test. More held-out seeds, successful baselines, object types and perturbations, as well as calibrated sensing and dynamic collision and force validation, are required before deployment. Depth Anything and wrist fusion have not demonstrated an improvement.
