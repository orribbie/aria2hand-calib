# Hand-eye calibration: Aria RGB camera to the YOR-v3 arm mount

This is the procedure that produced the calibration in use since 2026-09-15,
with the numbers it gave and the two things that had to change from the
package's documented recipe before any fit succeeded. Everything in this repository
is offline tooling, documentation and the resulting calibration; the collection itself moves the left
arm and is driven from the `yor-v3-calib` package.

What the calibration is: `T_C_B`, the rigid transform from the MuJoCo
arm-mount world frame (`B`: x forward, y left, z up, origin at the arm mount)
to the rectified Aria RGB camera frame (`C`: x right, y down, z forward).
Its inverse `T_B_C` has the camera centre in its last column.

## The marker and the numbers that worked

| Item | Value |
| --- | --- |
| ArUco dictionary | `DICT_4X4_250` |
| ArUco id | **0** |
| Marker size passed to the fitter | **65 mm** (outer black square, white margins excluded) |
| Where it sits | left-arm calibration mount (`stand_v2`), corner sites `left_R_U, L_U, L_B, R_B` = decoded corners TL, TR, BR, BL |
| Corners used for fitting | re-detected with the template refinement (`calib_tools.py refine`) |
| Fit | `yor-calib-fit --arm left --marker-size-mm 65 --fit-marker-mount` |
| Camera model | Aria Gen 2 RGB, rectified to a pinhole 2560 × 1920, fx = fy = 1110.2, cx = 1279.5, cy = 959.5, zero distortion |

Two points that cost a full evening, so they come first:

1. **65 mm, not 60.** The package README's worked example says the measured
   marker is 60 mm. With the marker on the arm now, 60 mm never fits: the
   samples cannot agree on any camera pose within the 3 px threshold. A size
   sweep on 280 poses puts the optimum at 65 mm (64 and 66 mm fit worse, 63
   and 67 fail). 65 mm is also the package default. Measure the black square
   if the print is ever replaced, and re-run `calib_tools.py sweep-size`.
2. **Refine the corners.** The auto collector's plain ArUco corners sit about
   6 px off the code cells on this marker (thin white rim on a dark mount).
   Collect with `--refine-template`, or re-detect afterwards from the saved
   candidate images with `calib_tools.py refine`. Nothing fits without it.

## Software

- `yor-v3-calib` at commit `55952b2` (billy's package,
  `https://github.com/NYU-robot-learning/yor-v3-calib`) **plus the local
  changes in `yor-v3-calib_local.patch`**: automatic collection with
  `--arm`, `--require-fresh`, `--refine-template`, the marker-mount correction
  (`--fit-marker-mount`), `--marker-size-mm`, and their tests. The untracked
  helper tools that go with it are copied in `yor-v3-calib-tools/`.
  To rebuild that checkout elsewhere:

  ```bash
  git clone https://github.com/NYU-robot-learning/yor-v3-calib.git && cd yor-v3-calib
  git checkout 55952b2
  git apply ~/aria2hand-calib/yor-v3-calib_local.patch
  cp ~/aria2hand-calib/yor-v3-calib-tools/* tools/
  python -m venv .venv && ./.venv/bin/pip install -e ".[live,dev]"
  ./.venv/bin/python -m pytest -q
  ```

  On Thor this checkout lives in `~/yor-v3-calib` with its `.venv` already
  built; all commands below use that interpreter.
- `aria_calib_pub.py` in this repo (same file as `demos/aria_calib_pub.py` in `orribbie/yor_odin_slam`): publishes the rectified Aria RGB
  stream the collector needs (`online_calib` and `images` on port 5555). Runs
  in the `aria2robot` conda env. The freshness patch is already applied to it.
- `calib_tools.py`: refine, combine, size sweep,
  leave-one-set-out, compare, show, crops montage, joint-offset diagnostic.
  Run it with the calib venv:

  ```bash
  ~/yor-v3-calib/.venv/bin/python ~/aria2hand-calib/calib_tools.py --help
  ```

## Hardware setup

- Aria Gen 2 on Thor's USB (`lsusb` shows `Oculus VR, Inc. Aria Gen 2`).
- Marker id 0 on the left calibration mount, oriented as the model's corner
  sites (see the table above). The camera, mount and lift must not move
  relative to each other afterwards; the model has a fixed arm mount and does
  not model the lift.
- Robot controller on the Pi (`pi-v3@100.98.224.40`, LAN `192.168.1.10`),
  RPC port 5557:

  ```bash
  cd ~/YOR-v3-aria && python robot/yor.py --hand wuji
  ```

  It homes the arms and the lift on start (lift at 0.625 m). The collector
  only reads joints and sends left-arm joint targets; it never homes or
  initialises the robot.

## Step 1: camera publisher (Thor, its own terminal, keep it running)

```bash
cd ~/aria2hand-calib && ~/miniconda3/envs/aria2robot/bin/python aria_calib_pub.py --check
```

`--check` connects, waits for one frame, prints the rectified `K` and the
frame size, and exits. Expected: `rgb 2560x1920, rectified K: fx=1110.2
fy=1110.2 cx=1279.5 cy=959.5`. Then start it for real:

```bash
cd ~/aria2hand-calib && ~/miniconda3/envs/aria2robot/bin/python aria_calib_pub.py
```

If the publisher is not running, `yor-calib-auto` hangs silently in
`fetch_intrinsics` (commlink subscribers wait forever). Nothing listening on
port 5555 is the first thing to check.

## Step 2: collect (moves the left arm)

Put the left arm where the marker faces the camera at 35 to 55 cm and the
right arm out of the way. Then, per dataset:

```bash
cd ~/yor-v3-calib && ./.venv/bin/yor-calib-auto --arm left --count 50 --aruco-id 0 \
  --camera-host 127.0.0.1 --robot-host 100.98.224.40 \
  --require-fresh --refine-template --output data/<name>
```

- The planner samples 50 valid poses within ±0.1 rad of the start pose, moves
  in 0.003 rad steps, waits until the joints are still, then captures.
- `--require-fresh` waits for a frame captured after the arm settled (the
  publisher stamps `capture_time_unix`); `--refine-template` stores refined
  corners directly. The 2026-09-15 sets were collected without either flag and
  were repaired offline (Step 3), so both are recommended, not required.
- The output directory must not exist. An aborted run leaves an empty one
  behind; `rmdir` it before retrying.
- **Collect at more than one wrist angle.** The fitted mount correction is
  only constrained by wrist (joint 7) diversity. The Sep 15 data has one set
  at joint 7 = 0° and five sets at 45° to 69° (the grasp posture). Fitting
  without the 0° set predicts that posture 10 px off; with it, every posture
  in the range is within 5.3 px.

Collections of 2026-09-15 (all 50 poses unless noted): `auto099` (wrist 0°),
`autonice` (63°), `autonice12` (51°), `autonice126` (50°), `autonice127`
(50°), `train_b` (30 poses, 51°, collected with both flags).

## Step 3: refine the corners offline

Skip if the set was collected with `--refine-template`. Otherwise:

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py refine data/<name>
```

This writes `data/<name>_refined` with the same joints and intrinsics and the
re-detected corners. It also reports "stored corners vs plain re-detection",
which must be 0 px: that proves the saved images and the stored arrays are
paired correctly. Have a look at the detections once:

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py crops data/<name>
```

The red polygon in `data/<name>/crops_montage.png` must hug the outer edge of
the black square.

## Step 4: combine the sets

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py combine data/today_all_refined \
  data/auto099_refined data/autonice_refined data/autonice12_refined \
  data/autonice126_refined data/autonice127_refined data/train_b
```

The combined set keeps a `source_set.npy` tag per pose for Step 6.

## Step 5: fit

```bash
cd ~/yor-v3-calib && ./.venv/bin/yor-calib-fit --arm left --data data/today_all_refined \
  --marker-size-mm 65 --fit-marker-mount
```

Expected for the Sep 15 data: `233/280 poses, mean inlier error 1.899 px`.
The output is `data/today_all_refined/hand_eye_calibration_left.npz`
(`T_C_B`, `T_eye_base` alias, `rvec`, `tvec`, `camera_matrix`,
`marker_size_m`, `marker_correction`, per-pose errors, inlier mask) and
`eye_to_base_left.npy` (`T_B_C`).

If the fit reports "No consistent camera pose", run the size sweep before
anything else:

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py sweep-size data/today_all_refined
```

## Step 6: check it

Leave-one-set-out on the combined set (fit on five sets, test on the sixth):

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py loo data/today_all_refined
```

Held-out check of a saved calibration against a set it was not fitted on
(writes overlays into `<data>/validation/`):

```bash
cd ~/yor-v3-calib && ./.venv/bin/python tools/validate_calibration.py \
  --data data/<holdout>_refined --calibration data/today_all_refined/hand_eye_calibration_left.npz
```

Sep 15 results, in-sample per set (median / worst pose): auto099 2.27 / 5.17 px,
autonice 2.41 / 5.31, autonice12 1.94 / 4.41, autonice126 1.91 / 4.91,
autonice127 2.06 / 4.59, train_b 2.22 / 4.23. Leave-one-set-out: every
pitched-wrist set is predicted by the others at 1.9 to 3.5 px median; the 0°
set is 10.6 px when left out, which is why it must be in the fit. At the 40 cm
working distance 3 px is about 1 mm.

## Step 7: use the result

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py show data/today_all_refined/hand_eye_calibration_left.npz
```

Calibration of 2026-09-15 (`data/today_all_refined`, 280 poses, 65 mm marker,
mount correction 10.8, 5.6, −5.1 mm at 3.6°):

`T_C_B` (arm-mount world → camera):

```
[[-0.999849, -0.003145, -0.017089,  0.069407],
 [ 0.013357,  0.489944, -0.871651,  0.558646],
 [ 0.011114, -0.871748, -0.489828,  0.236901],
 [ 0.0,       0.0,       0.0,       1.0     ]]
```

`T_B_C` (camera → arm-mount world, camera centre in the last column):

```
[[-0.999849,  0.013357,  0.011114,  0.059302],
 [-0.003145,  0.489944, -0.871748, -0.066969],
 [-0.017089, -0.871651, -0.489828,  0.604171],
 [ 0.0,       0.0,       0.0,       1.0     ]]
```

OpenCV form: `rvec = [-0.009244, -2.697431, 1.578288]` rad,
`tvec = [0.069407, 0.558646, 0.236901]` m. The camera sits 5.9 cm forward,
6.7 cm right and 60.4 cm up from the arm-mount origin, looking forward and
pitched down about 29°.

Compared with the previous calibration (`combined60_train`, 2026-09-11,
60 mm marker, 41/55 at 1.88 px) the camera moved 9.9 mm and 0.96°, which is
2 to 6 px at grasp distance. The old file predicts the Sep 15 poses 14 to
17 px off, mostly because the marker mount now sits differently, not because
the camera moved. To quantify a change yourself:

```bash
cd ~/yor-v3-calib && ./.venv/bin/python ~/aria2hand-calib/calib_tools.py compare \
  data/today_all_refined/hand_eye_calibration_left.npz data/combined60_train/hand_eye_calibration_left.npz \
  --data data/autonice126_refined
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `yor-calib-auto` hangs after reading the joints, traceback ends in `recv_multipart` on Ctrl-C | No camera publisher on port 5555. Start `aria_calib_pub.py`. |
| `FileExistsError` on the output directory | Left over from an aborted run. `rmdir data/<name>`. |
| `No consistent camera pose for at least 60% of calibration samples` | In this order: corners not refined (Step 3), wrong `--marker-size-mm` (sweep it), missing `--fit-marker-mount`, marker mount moved or loose. |
| `Mount fit has fewer than 60% inliers` | Same causes; also fewer than 12 varied poses or a single wrist angle. |
| Fit succeeds but the held-out error is 10 px or more | The held-out posture is outside the wrist range of the fitted sets. Add a set at that wrist angle. |
| Old calibration suddenly 15 to 20 px off | Marker mount re-seated or camera moved. Recalibrate; `compare` tells you which. |
| yor.py aborts with `Assertion failed: pfd.revents & POLLIN (src/signaler.cpp)` after Ctrl-C | libzmq assertion while the RPC server shuts down; harmless, restart yor.py. |

## Files in this repository

- `calibration.md` this document
- `calib_tools.py` offline helpers described above
- `test_calib_tools.py` pytest for the helpers on a synthetic dataset
  (`cd ~/aria2hand-calib && ~/yor-v3-calib/.venv/bin/python -m pytest test_calib_tools.py`)
- `yor-v3-calib_local.patch` the local changes to the calib package on top of `55952b2`
- `yor-v3-calib-tools/` the package's untracked tools: `collect_wide.py`,
  `validate_calibration.py`, `aria_publisher_freshness.patch`

- `aria_calib_pub.py` rectified Aria RGB publisher for the collector (Step 1)
- `results/2026-09-15_today_all_refined/` the calibration in use:
  `hand_eye_calibration_left.npz` (T_C_B, marker correction, per-pose errors),
  `eye_to_base_left.npy` (T_B_C), and the combined 280-pose dataset it was fitted
  on (joints, refined corners, intrinsics, per-pose source tag; no images), so
  the fit, `loo` and `sweep-size` can be re-run from this repo alone:

  ```bash
  cd ~/aria2hand-calib && ~/yor-v3-calib/.venv/bin/python calib_tools.py show results/2026-09-15_today_all_refined/hand_eye_calibration_left.npz
  ```

The raw collections (candidate images, per-set folders) stay in
`~/yor-v3-calib/data/` on Thor; they are not in any repository.
