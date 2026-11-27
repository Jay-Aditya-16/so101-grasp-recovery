# Offline transfer IK diagnostic

The local seed-1030 Phase II trace was replayed kinematically at 15.25 s, the logged transfer-planning failure. The script reconstructs the MuJoCo model and recorded `qpos` and `qvel` and solves for the existing box-center target `[0.05, 0.22, 0.12]` m. It performs no motion or physics rollout. The two original seeds (current and home) were tried with the existing downward and position-only objectives, plus eight explicit orientations on a 40-degree cone around vertical. These are targeted diagnostic candidates, not an exhaustive ablation or proof of reachability.

| Solver input | Position error | Downward-axis dot | Existing pose gate | Sampled arm path |
|---|---:|---:|---|---|
| Downward, current seed | 119.83 mm | -0.979 | fail | not checked |
| Downward, home seed | 86.48 mm | 0.833 | fail | not checked |
| Position only, current seed | 0.690 mm | 0.630 | fail | not checked |
| Position only, home seed | 0.702 mm | 0.261 | fail | not checked |
| 40-degree cone, +Y direction, current seed | 3.41 mm | 0.766 | pass | no penetration detected |

The existing relaxed pose gate requires no more than 5 mm position error and dot at least 0.70. Thus, in this recorded state, the original four attempts fail, while one additional direction satisfies that mathematical gate and the existing 101-sample path check. This identifies a solver search bottleneck, not a successful placement. A second offline check rigidly attaches the recorded cube pose to the tool and found no cube-environment penetration over those 101 samples. This uses simulator ground truth and an ideal no-slip grasp; it ignores cube-robot contact and cannot check slip, force, velocity, torque, or continuous swept volume. Neither check establishes physical safety. This result does not establish physical safety or warrant use on the SO-101 hardware. See `reports/phase3-ik-diagnostic/seed1030.json` for all 20 attempts.

Reproduce on the configured machine:

```sh
PYTHONPATH=/private/tmp/so101-lerobot-public061 /Users/jayaditya/so101-lerobot-juxi/.venv/bin/python scripts/diagnose_transfer_ik.py \
 datasets/local-rollouts/phase3-v2-phase2-1030.npz --tutorial ../so101-physicalai-tutorial \
 --output outputs/phase3-ik-diagnostic/seed1030.json
```

The reconstruction uses recorded simulator state and fixed tutorial geometry. It is an offline analysis tool, not a deployable camera-based controller. The recorded trace is local; the compact diagnostic output is versioned.
