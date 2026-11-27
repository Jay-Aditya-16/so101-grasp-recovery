# Phase III decision maths and research basis

## Scope and assumptions

This experiment selects among six pre-existing box placement targets after a suspected grasp failure and scripted recovery. It does not select a grasp, learn a control policy, or update ACT weights. The context `x = [1, (cube_x-.20)/.10, cube_y/.18]` comes from a camera estimate. The task outcome is a binary simulated evaluation label after release, not an online observation of object position. Feasibility depends on the current robot state and is checked by the existing IK and discretely sampled path checker; it is not guaranteed safe in physics or on hardware.

## Outcome-conditioned preference

For each candidate `a`, use independent ridge regression with regularization λ=1 and **centered** reward `y=2r-1`, with `r=1` for verified placement and `r=0` for a verified completed failure:

`A_a = I + Σ_(i:a_i=a) x_i x_iᵀ`, `b_a = Σ_(i:a_i=a) x_i y_i`.

Compute `theta_a = A_a⁻¹ b_a` by a linear solve; rank with `s_a(x) = xᵀ theta_a + α sqrt(xᵀ A_a⁻¹ x)` and `α=.25`. Try candidates by this rank, retaining the existing feasibility veto. Scores are not success probabilities; the exploration coefficient is a heuristic, not a validated confidence interval or a safety guarantee. The method is inspired by [LinUCB](https://proceedings.mlr.press/v15/chu11a/chu11a.pdf), without claiming its assumptions or bounds hold here. Logged historical outcomes remain binary; their transformation occurs when fitting, so no recorded data are rewritten.

The centered encoding makes a verified failure informative in its own context. If no prior data exist and a failure occurs at `x`, set `v=xᵀx>0`. By Sherman–Morrison, its new mean is `-v/(1+v)` and uncertainty is `sqrt(v/(1+v))`, whereas its initial mean was zero and uncertainty `sqrt(v)`. Therefore for α≥0 its score strictly falls below its original score and below a still-untried candidate at that same context. This does **not** prove avoidance in a different context, one-shot learning of a grasp, or future success. The linear model can generalize in the wrong direction, particularly across distant contexts; this must be measured.

## Missing outcomes and evaluation

Interruptions, no candidate selection and unfinished release are censored, not assigned a failure reward. Consequently the learning data are selected from trials that reached release; they cannot establish how a method performs on failed grasps or on all started trials. Log counts of every started trial, selections, completed labels, abstentions and task successes. Update only after a completed episode; the system is not doing within-attempt weight updates. Training memory and held-out evaluation must stay separate. Compare frozen memory with Phase II and simple alternatives on matched unseen seeds and identical simulation settings; report all trials and uncertainty, not only labeled ones. Deterministic action choices lack randomized propensities, so the existing logs alone cannot support counterfactual or causal claims.

## Research choices

[BORGES](https://proceedings.mlr.press/v155/danielczuk21a.html) explores structured grasp-pose candidates using grasp outcomes. Our current targets are placement targets and are not a reproduction. [HIL-SERL](https://hil-serl.github.io/) combines demonstrations, human interventions and online reinforcement learning; it is a later reference for grasp-recovery experiments, not a drop-in ACT update. [V-JEPA 2](https://arxiv.org/abs/2506.09985) uses a video representation and a separately action-conditioned predictor for robot planning. The [official implementation](https://github.com/facebookresearch/vjepa2) distinguishes visual encoders from action-conditioned planning models; its requirements and latency on this M1/L4 setup are unverified. [LeWorldModel](https://github.com/lucas-maes/le-wm) offers a smaller research direction, but has not been trained or tested on our task. A world model would need action-aligned two-view trajectories and held-out prediction and control comparisons before influencing robot motion. None of these models establishes consciousness, guaranteed physical understanding or immediate lifelong learning.

The red centroid is projected onto a fixed-height plane using the tutorial camera geometry; errors in height, visible side faces, camera calibration or occlusion can bias XY. Relative monocular depth is not metric distance by itself. The existing success proxy requires a lifted cube, support, release, footprint and low **linear** velocity; it does not check angular velocity. These limitations affect both training labels and evaluation. The existing path sampler also does not model the carried object's swept volume, torque or velocity limits; no physical safety conclusion follows.

## Pilot and independent arithmetic check

One predeclared paired smoke test used seed 1030, the 20,000-step checkpoint and a 25-second horizon. This seed was absent from the two-record bandit memory. Both Phase II and frozen Phase III lifted the cube by approximately 97.6 mm but failed to place it: the existing planner found no candidate that passed its pose constraints, so Phase III selected no placement candidate and no label was added. This is one case and cannot establish either effectiveness or equivalence. The fixed-order Phase II trial was run first; the order was not randomized, another limitation. See `reports/phase3-v2/` for the plan, trial summaries, manifest and checks, and the two `datasets/local-rollouts/phase3-v2-*.npz` files locally for images and transitions. Both traces passed the existing alignment validator. The frozen memory's before/after hashes matched.

MATLAB independently computed the centered-reward scores for explicitly **synthetic** success and failure records; the largest absolute difference versus Python for mean, uncertainty or score was 4.45e-16. At the failure context, the failed center action has mean -0.5. This verifies arithmetic for these inputs, not the robot physics, reward quality or real-world performance. The synthetic records are not robot experience and are not used as training memory.
