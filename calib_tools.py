#!/usr/bin/env python3
"""Helpers around the yor-v3-calib package that made the 2026-09-15 hand-eye
calibration work (see calibration.md next to this file).

Everything here is offline: it reads collected datasets and calibration files,
never talks to the robot or the camera. The only moving-robot step is the
collection itself, which stays in yor-v3-calib (`yor-calib-auto`).

Sub-commands

  refine        re-detect the marker corners of a collected dataset from its
                saved candidate images with the template refinement on, and
                write a `<data>_refined` copy. The auto collector's plain
                corners sit ~6 px off the code cells on our marker; this is
                the single change that turned "No consistent camera pose"
                into a 1.6 px fit.
  combine       concatenate several (refined) datasets into one, keeping a
                per-sample source tag for leave-one-set-out checks.
  sweep-size    fit at several marker sizes and print inliers / error per
                size, camera-only and with the mount correction. 65 mm is
                what our marker measures in this test; 60 mm never fits.
  loo           leave-one-set-out: fit on all sets but one, test on the
                held-out set. Needs a `combine`d dataset.
  compare       difference between two calibrations (camera shift, rotation,
                pixel effect at grasp depth, reprojection on a dataset).
  show          print T_C_B, T_B_C, rvec/tvec and the fit statistics of a
                calibration file.
  crops         montage of the detected corners on the saved images, for a
                visual sanity check of the detections.
  joint-offsets diagnostic only: does a per-joint zero offset explain the
                residuals? Not used for the calibration itself.

Run with the yor-v3-calib virtualenv (opencv-contrib with ArUco, mujoco,
scipy):

    ~/yor-v3-calib/.venv/bin/python calib/calib_tools.py refine data/autonice126

`--calib-repo` (default ~/yor-v3-calib) locates the package if it is not
installed in the interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ARUCO_ID = 0                 # DICT_4X4_250, id 0 -- the marker on the left calibration mount
MARKER_SIZE_MM = 65.0        # outer black square that fitted on 2026-09-15 (60 mm never did)
THRESHOLD_PX = 3.0           # the fitter's inlier threshold


# --------------------------------------------------------------------------- #
# package location
# --------------------------------------------------------------------------- #

def _import_calib(repo: str | None):
    """Import yor_v3_calib, falling back to <repo>/src on sys.path."""
    try:
        import yor_v3_calib  # noqa: F401
    except ImportError:
        src = Path(repo or "~/yor-v3-calib").expanduser() / "src"
        if not (src / "yor_v3_calib").is_dir():
            sys.exit(f"yor_v3_calib not importable and {src} has no package; pass --calib-repo")
        sys.path.insert(0, str(src))
    from yor_v3_calib import calibration as C
    from yor_v3_calib import live as L
    return C, L


# --------------------------------------------------------------------------- #
# dataset helpers
# --------------------------------------------------------------------------- #

def load_dataset(d: Path, arm: str = "left") -> dict:
    d = Path(d)
    out = {
        "dir": d,
        "q": np.load(d / f"joint_angles_{arm}.npy"),
        "uv": np.load(d / f"two_d_coordinates_{arm}.npy"),
    }
    with np.load(d / "camera_intrinsics.npz") as f:
        out["K"] = f["camera_matrix"]
        out["D"] = f["dist_coeffs"] if "dist_coeffs" in f else np.zeros(5)
    src = d / "source_set.npy"
    out["src"] = np.load(src) if src.exists() else np.array([d.name] * len(out["q"]))
    if out["q"].shape[0] != out["uv"].shape[0]:
        sys.exit(f"{d}: {len(out['q'])} joint rows but {len(out['uv'])} corner rows")
    return out


def sample_to_candidate(d: Path, n: int) -> dict:
    """Map sample index (0-based) -> candidate image number, via collection_log.json.

    The auto collector numbers images by candidate pose, and a candidate only
    becomes a sample if the marker was found and the arm did not drift, so the
    two numberings differ. Without a log the images are assumed to be in
    sample order (manual collection, `candidate_001.png` = sample 1).
    """
    log = Path(d) / "collection_log.json"
    if not log.exists():
        return {k: k + 1 for k in range(n)}
    entries = json.loads(log.read_text())
    m = {int(e["sample"]) - 1: int(e["candidate"]) for e in entries if e.get("sample") is not None}
    if len(m) != n:
        sys.exit(f"{d}: log has {len(m)} samples but the arrays have {n}")
    return m


def camera_centre(T_C_B: np.ndarray) -> np.ndarray:
    return -T_C_B[:3, :3].T @ T_C_B[:3, 3]


def reproj_errors(C, T_C_B, xyz, uv, K, D):
    import cv2
    p = cv2.projectPoints(xyz.reshape(-1, 3), cv2.Rodrigues(T_C_B[:3, :3])[0], T_C_B[:3, 3], K, D)[0]
    return np.linalg.norm(p.reshape(uv.shape) - uv, axis=2).mean(axis=1)


# --------------------------------------------------------------------------- #
# refine
# --------------------------------------------------------------------------- #

def cmd_refine(a):
    import cv2
    C, L = _import_calib(a.calib_repo)
    d = Path(a.data)
    ds = load_dataset(d, a.arm)
    out = Path(a.out) if a.out else d.with_name(d.name + "_refined")
    if out.exists() and not a.force:
        sys.exit(f"{out} exists; pass --force to overwrite")
    cand = sample_to_candidate(d, len(ds["q"]))
    keep, refined, plain_dev = [], [], []
    for k in range(len(ds["q"])):
        img = cv2.imread(str(d / f"candidate_{cand[k]:03d}.png"))
        if img is None:
            sys.exit(f"{d}: missing candidate_{cand[k]:03d}.png for sample {k + 1}")
        plain = L.aruco_corners_for_id(img, a.aruco_id, refine_template=False)
        ref = L.aruco_corners_for_id(img, a.aruco_id, refine_template=True)
        if plain is not None:
            plain_dev.append(float(np.linalg.norm(plain - ds["uv"][k], axis=1).mean()))
        if ref is None:
            print(f"  sample {k + 1}: refinement failed, sample dropped", flush=True)
            continue
        keep.append(k)
        refined.append(ref)
    if len(keep) < 6:
        sys.exit(f"only {len(keep)} samples refined; nothing to fit")
    refined = np.asarray(refined, dtype=np.float64)
    shift = np.linalg.norm(refined - ds["uv"][keep], axis=2).mean(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("camera_intrinsics.npz", f"initial_{a.arm}_q.npy", "initial_rgb.png",
              "initial_left_q.npy", "initial_right_q.npy"):
        if (d / f).exists():
            shutil.copy2(d / f, out / f)
    np.save(out / f"joint_angles_{a.arm}.npy", ds["q"][keep])
    np.save(out / f"two_d_coordinates_{a.arm}.npy", refined)
    settings = {}
    if (d / "settings.json").exists():
        settings = json.loads((d / "settings.json").read_text())
    settings.update({
        "refine_template": True, "source": str(d), "aruco_id": a.aruco_id,
        "note": "corners re-detected offline from the saved candidate images with "
                "refine_template=True (calib_tools.py refine); joints and intrinsics unchanged",
        "samples_dropped": int(len(ds["q"]) - len(keep)),
    })
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    pd = f"max {max(plain_dev):.2f} px" if plain_dev else "n/a"
    print(f"{d.name}: {len(keep)}/{len(ds['q'])} samples refined -> {out}")
    print(f"  stored corners vs plain re-detection: {pd} (0 = the images and the arrays match)")
    print(f"  refined vs stored: median {np.median(shift):.2f} px, max {shift.max():.2f} px")


# --------------------------------------------------------------------------- #
# combine
# --------------------------------------------------------------------------- #

def cmd_combine(a):
    sets = [load_dataset(Path(p), a.arm) for p in a.data]
    K0 = sets[0]["K"]
    for s in sets[1:]:
        if not np.allclose(s["K"], K0):
            sys.exit(f"{s['dir']}: intrinsics differ from {sets[0]['dir']}; refusing to combine")
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} exists; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    q = np.vstack([s["q"] for s in sets])
    uv = np.vstack([s["uv"] for s in sets])
    src = np.concatenate([np.array([s["dir"].name] * len(s["q"])) for s in sets])
    np.save(out / f"joint_angles_{a.arm}.npy", q)
    np.save(out / f"two_d_coordinates_{a.arm}.npy", uv)
    np.save(out / "source_set.npy", src)
    np.savez(out / "camera_intrinsics.npz", camera_matrix=K0, dist_coeffs=sets[0]["D"])
    (out / "settings.json").write_text(json.dumps({
        "arm": a.arm, "sources": [str(s["dir"]) for s in sets], "samples": int(len(q)),
        "note": "concatenation by calib_tools.py combine; fit with "
                f"`yor-calib-fit --arm {a.arm} --data {out} --marker-size-mm {MARKER_SIZE_MM:.0f} --fit-marker-mount`",
    }, indent=2))
    j7 = np.degrees(q[:, 6])
    print(f"{out}: {len(q)} samples from {len(sets)} sets; wrist joint 7 range "
          f"[{j7.min():+.0f}, {j7.max():+.0f}] deg")


# --------------------------------------------------------------------------- #
# fits shared by sweep-size and loo
# --------------------------------------------------------------------------- #

def _fit(C, q, uv, K, D, size_mm, mount, thr, arm):
    xyz = C.marker_points(q, arm, marker_size_m=size_mm / 1000.0)
    if mount:
        T, e, m, mc = C.estimate_extrinsics_and_mount(xyz, uv, K, D, threshold_px=thr)
    else:
        T, e, m = C.estimate_extrinsics(xyz, uv, K, D, threshold_px=thr)
        mc = None
    return T, e, m, mc


def cmd_sweep_size(a):
    C, _ = _import_calib(a.calib_repo)
    ds = load_dataset(Path(a.data), a.arm)
    print(f"{ds['dir']}: {len(ds['q'])} samples, threshold {a.threshold_px} px")
    print(f"{'size mm':>8} | {'camera only':>18} | {'camera + mount':>18}")
    for size in a.sizes:
        cells = []
        for mount in (False, True):
            try:
                _, e, m, _ = _fit(C, ds["q"], ds["uv"], ds["K"], ds["D"], size, mount, a.threshold_px, a.arm)
                cells.append(f"{m.sum():>3}/{len(m):<3} {e[m].mean():5.2f} px")
            except ValueError as ex:
                cells.append(f"FAIL ({str(ex)[:9]})")
        print(f"{size:>8.1f} | {cells[0]:>18} | {cells[1]:>18}")


def cmd_loo(a):
    C, _ = _import_calib(a.calib_repo)
    ds = load_dataset(Path(a.data), a.arm)
    names = list(dict.fromkeys(ds["src"].tolist()))
    if len(names) < 2:
        sys.exit("leave-one-set-out needs a dataset made by `combine` (source_set.npy with >= 2 sets)")
    print(f"{ds['dir']}: {len(ds['q'])} samples in {len(names)} sets; marker {a.marker_size_mm} mm, "
          f"threshold {a.threshold_px} px, mount correction {'on' if not a.no_mount else 'off'}")
    for name in names:
        te = ds["src"] == name
        tr = ~te
        try:
            T, e, m, mc = _fit(C, ds["q"][tr], ds["uv"][tr], ds["K"], ds["D"], a.marker_size_mm,
                               not a.no_mount, a.threshold_px, a.arm)
        except ValueError as ex:
            print(f"  held out {name:<22} n={te.sum():>3}: training fit FAILED ({ex})")
            continue
        xyz = C.marker_points(ds["q"][te], a.arm, marker_size_m=a.marker_size_mm / 1000.0)
        if mc is not None:
            xyz = C.correct_marker_points(xyz, mc)
        et = reproj_errors(C, T, xyz, ds["uv"][te], ds["K"], ds["D"])
        print(f"  held out {name:<22} n={te.sum():>3}: median {np.median(et):5.2f} px  "
              f"max {et.max():5.2f} px  <{a.threshold_px:.0f}px {(et < a.threshold_px).sum()}/{te.sum()}")


# --------------------------------------------------------------------------- #
# show / compare
# --------------------------------------------------------------------------- #

def _load_calibration(path: str):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    return out


def _print_matrix(name, T):
    print(f"{name} =")
    for row in T:
        print("  [" + ", ".join(f"{v:+.6f}" for v in row) + "]")


def cmd_show(a):
    import cv2
    z = _load_calibration(a.calibration)
    T = z["T_C_B"]
    print(f"{a.calibration}")
    print(f"arm {z.get('arm')}  marker {float(z['marker_size_m']) * 1000:.0f} mm  method {z.get('calibration_method', '?')}")
    e, m = z["pose_errors_px"], z["inliers"]
    print(f"inliers {m.sum()}/{len(m)}  mean inlier error {e[m].mean():.3f} px  worst inlier {e[m].max():.2f} px")
    _print_matrix("T_C_B (arm-mount world -> camera)", T)
    _print_matrix("T_B_C (camera -> arm-mount world; last column = camera centre)", np.linalg.inv(T))
    print("rvec (rad):", np.round(z["rvec"].ravel(), 6).tolist(), " tvec (m):", np.round(z["tvec"].ravel(), 6).tolist())
    if "marker_correction" in z:
        mc = z["marker_correction"]
        rot = np.degrees(np.linalg.norm(cv2.Rodrigues(mc[:3, :3])[0]))
        print(f"marker mount correction: t (mm) {np.round(mc[:3, 3] * 1000, 1).tolist()}  rotation {rot:.1f} deg")


def cmd_compare(a):
    import cv2
    A = _load_calibration(a.a)
    if a.b_json:
        B = {"T_C_B": np.asarray(json.loads(a.b_json), dtype=float).reshape(4, 4)}
    else:
        B = _load_calibration(a.b)
    Ta, Tb = A["T_C_B"], B["T_C_B"]
    ca, cb = camera_centre(Ta), camera_centre(Tb)
    rot = np.degrees(np.linalg.norm(cv2.Rodrigues(Ta[:3, :3].T @ Tb[:3, :3])[0]))
    print(f"A: {a.a}\nB: {a.b or 'literal matrix'}")
    print(f"camera centre A {np.round(ca, 4).tolist()}  B {np.round(cb, 4).tolist()}")
    print(f"camera shift B-A (mm, arm-mount frame): {np.round((cb - ca) * 1000, 1).tolist()}  |{np.linalg.norm(cb - ca) * 1000:.1f} mm|")
    print(f"relative rotation: {rot:.2f} deg")
    K = A["camera_matrix"] if "camera_matrix" in A else None
    if K is not None:
        for depth in a.depths:
            pc = np.array([0.0, 0.0, depth, 1.0])          # a point on B's optical axis
            pw = np.linalg.inv(Tb) @ pc
            pa = Ta @ pw
            dpx = K @ (pa[:3] / pa[2]) - K @ (pc[:3] / pc[2])
            print(f"  point {depth:.2f} m ahead: {np.linalg.norm(pa[:3] - pc[:3]) * 1000:.1f} mm apart in camera coords, "
                  f"{np.linalg.norm(dpx[:2]):.1f} px in the image")
    if a.data:
        C, _ = _import_calib(a.calib_repo)
        ds = load_dataset(Path(a.data), a.arm)
        for label, cal in (("A", A), ("B", B)):
            size = float(cal["marker_size_m"]) if "marker_size_m" in cal else MARKER_SIZE_MM / 1000.0
            xyz = C.marker_points(ds["q"], a.arm, marker_size_m=size)
            if "marker_correction" in cal:
                xyz = C.correct_marker_points(xyz, cal["marker_correction"])
            e = reproj_errors(C, cal["T_C_B"], xyz, ds["uv"], ds["K"], ds["D"])
            print(f"  {label} on {ds['dir'].name}: median {np.median(e):.2f} px  max {e.max():.2f} px  "
                  f"<{THRESHOLD_PX:.0f}px {(e < THRESHOLD_PX).sum()}/{len(e)}")


# --------------------------------------------------------------------------- #
# crops montage
# --------------------------------------------------------------------------- #

def cmd_crops(a):
    import cv2
    d = Path(a.data)
    ds = load_dataset(d, a.arm)
    cand = sample_to_candidate(d, len(ds["q"]))
    idx = [s - 1 for s in a.samples] if a.samples else list(np.linspace(0, len(ds["q"]) - 1, 6).astype(int))
    tiles = []
    for k in idx:
        img = cv2.imread(str(d / f"candidate_{cand[k]:03d}.png"))
        if img is None:
            sys.exit(f"missing candidate_{cand[k]:03d}.png")
        c = ds["uv"][k]
        cx, cy = c.mean(axis=0).astype(int)
        r = a.radius
        x0, y0 = max(0, cx - r), max(0, cy - r)
        crop = img[y0:cy + r, x0:cx + r].copy()
        pts = (c - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(crop, [pts], True, (0, 0, 255), 1)
        crop = cv2.resize(crop, (a.tile, a.tile), interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"s{k + 1} c{cand[k]}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(crop)
    out = Path(a.out) if a.out else d / "crops_montage.png"
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"wrote {out} ({len(tiles)} tiles); the red polygon must hug the outer edge of the black square")


# --------------------------------------------------------------------------- #
# joint-offset diagnostic
# --------------------------------------------------------------------------- #

def cmd_joint_offsets(a):
    import cv2
    from scipy.optimize import least_squares
    C, _ = _import_calib(a.calib_repo)
    ds = load_dataset(Path(a.data), a.arm)
    q, uv, K, D = ds["q"], ds["uv"], ds["K"], ds["D"]
    size = a.marker_size_mm / 1000.0
    lim = np.radians(a.max_offset_deg)

    def se3(x):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], float))[0]
        T[:3, 3] = x[3:6]
        return T

    def fit(joints, mount):
        T0, _, _ = C.estimate_extrinsics(C.marker_points(q, a.arm, marker_size_m=size), uv, K, D, threshold_px=25.0)
        n_j, n_m = len(joints), (6 if mount else 0)
        x0 = np.concatenate([cv2.Rodrigues(T0[:3, :3])[0].ravel(), T0[:3, 3], np.zeros(n_j + n_m)])
        lb = np.concatenate([np.full(6, -np.inf), np.full(n_j, -lim), np.full(n_m, -np.inf)])
        ub = -lb

        def resid(x):
            qq = q.copy()
            for i, j in enumerate(joints):
                qq[:, j] += x[6 + i]
            try:
                xyz = C.marker_points(qq, a.arm, marker_size_m=size)
            except ValueError:
                return np.full(uv.size, 1e3)
            if mount:
                xyz = C.correct_marker_points(xyz, se3(x[6 + n_j:]))
            p = cv2.projectPoints(xyz.reshape(-1, 3), x[:3], x[3:6], K, D)[0].reshape(-1, 2)
            return (p - uv.reshape(-1, 2)).ravel()

        r = least_squares(resid, x0, bounds=(lb, ub), loss="huber", f_scale=4.0, max_nfev=400)
        err = np.linalg.norm(r.fun.reshape(-1, 4, 2), axis=2).mean(axis=1)
        offs = np.degrees(r.x[6:6 + n_j])
        tag = f"joints {[j + 1 for j in joints]}" + (" +mount" if mount else "")
        print(f"  {tag:<34} median {np.median(err):5.2f} px  <{THRESHOLD_PX:.0f}px {(err < THRESHOLD_PX).sum()}/{len(err)}  "
              f"offsets(deg) {np.round(offs, 2).tolist()}")

    print(f"{ds['dir']}: {len(q)} samples, marker {a.marker_size_mm} mm. DIAGNOSTIC ONLY: the fitter cannot "
          f"store joint offsets, and 50 samples with +-5 deg of motion overfit them easily.")
    fit([], False)
    fit([], True)
    for j in range(7):
        fit([j], False)
    fit(list(range(4)), False)
    fit(list(range(7)), False)
    fit(list(range(7)), True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-repo", default=None, help="yor-v3-calib checkout (default ~/yor-v3-calib)")
    p.add_argument("--arm", default="left", choices=("left", "right"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("refine", help="re-detect corners with the template refinement -> <data>_refined")
    s.add_argument("data")
    s.add_argument("--out", default=None)
    s.add_argument("--aruco-id", type=int, default=ARUCO_ID)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_refine)

    s = sub.add_parser("combine", help="concatenate datasets into one")
    s.add_argument("out")
    s.add_argument("data", nargs="+")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_combine)

    s = sub.add_parser("sweep-size", help="inliers / error per marker size")
    s.add_argument("data")
    s.add_argument("--sizes", type=float, nargs="+", default=[58, 60, 62, 64, 65, 66, 68, 70])
    s.add_argument("--threshold-px", type=float, default=THRESHOLD_PX)
    s.set_defaults(fn=cmd_sweep_size)

    s = sub.add_parser("loo", help="leave-one-set-out on a combined dataset")
    s.add_argument("data")
    s.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    s.add_argument("--threshold-px", type=float, default=THRESHOLD_PX)
    s.add_argument("--no-mount", action="store_true")
    s.set_defaults(fn=cmd_loo)

    s = sub.add_parser("show", help="print T_C_B / T_B_C and fit statistics")
    s.add_argument("calibration")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("compare", help="difference between two calibrations")
    s.add_argument("a", help="calibration npz")
    s.add_argument("b", nargs="?", default=None, help="calibration npz (or use --b-json)")
    s.add_argument("--b-json", default=None, help="4x4 T_C_B as a JSON list, instead of a file")
    s.add_argument("--data", default=None, help="dataset to reproject both calibrations on")
    s.add_argument("--depths", type=float, nargs="+", default=[0.35, 0.45, 0.55])
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("crops", help="montage of detected corners on the saved images")
    s.add_argument("data")
    s.add_argument("--samples", type=int, nargs="*", default=None, help="1-based sample numbers (default: 6 spread out)")
    s.add_argument("--radius", type=int, default=150)
    s.add_argument("--tile", type=int, default=400)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_crops)

    s = sub.add_parser("joint-offsets", help="diagnostic: per-joint zero offsets vs residuals")
    s.add_argument("data")
    s.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    s.add_argument("--max-offset-deg", type=float, default=10.0)
    s.set_defaults(fn=cmd_joint_offsets)

    a = p.parse_args(argv)
    if a.cmd == "compare" and not a.b and not a.b_json:
        p.error("compare needs a second calibration file or --b-json")
    a.fn(a)


if __name__ == "__main__":
    main()
