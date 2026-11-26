# Monocular depth experiment

We evaluated [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, on the overhead images from the four existing ACT traces. Its output is relative depth, not metres. The red pixel mask is a scene-specific evaluation aid. Ground-truth depth was rendered independently from saved MuJoCo states for scoring only; neither it nor the model output controls the arm. Code is in `scripts/evaluate_monocular_depth.py`; structured results are in `reports/phase2-monocular-depth*.json`.

At the initial frame, all four predicted the red cube closer than its surrounding ring; the simulator's median depth difference was approximately 30 mm. Relative-score differences were 0.023–0.042 in arbitrary units, and CPU inference on the M1 was 0.50–0.77 seconds per image (excluding setup). This is four nearly identical cube scenes, not a depth benchmark or metric calibration.

At the monitor alerts, the cube was mostly or fully obscured. The surrounding ring included the nearby gripper and was closer than the visible red pixels in simulator depth for all three partially visible cases. The model ranked the ring closer in one case and the red region closer in two; the fourth had no visible red pixels. Thus the nearby ring is no longer a table-only comparison, and the model did not reliably rank these regions under occlusion. Monocular depth cannot recover a hidden object's position by itself. MuJoCo emitted a depth accuracy warning on this Mac, so simulator reference values are approximate.

The experiment does not validate metric distance, depth at arbitrary scales, temporal tracking, or real-world transfer. The 60-pixel visibility gate for control remains unchanged. A useful next step is to evaluate synchronized wrist images and recover an unobstructed observation in simulation, then calibrate any metric depth using independent known references and evaluate held-out objects and views before using it for grasp candidates.

To reproduce on the local environment, install `transformers==4.46.3` and dependencies into a separate environment containing PyTorch, torchvision, MuJoCo, OpenCV and NumPy. Run:

```sh
python scripts/evaluate_monocular_depth.py --tutorial ../so101-physicalai-tutorial \
  --output reports/depth.json datasets/local-rollouts/phase1-act-seed1001.npz
```

Add `--frame alert` for the first monitor event. The first run downloads model weights. The raw trace files are local and excluded from Git.

Reference: [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414) and [official code](https://github.com/DepthAnything/Depth-Anything-V2).
