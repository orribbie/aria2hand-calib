"""Tests for calib_tools.py on a synthetic dataset: marker images rendered from
the arm model and a known camera pose, so every sub-command has a ground truth.

Run with the yor-v3-calib virtualenv (opencv-contrib ArUco, mujoco, scipy):

    ~/yor-v3-calib/.venv/bin/python -m pytest calib/test_calib_tools.py -q
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CALIB_REPO = os.environ.get("YOR_CALIB_REPO", str(Path("~/yor-v3-calib").expanduser()))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")
if not hasattr(cv2, "aruco"):
    pytest.skip("opencv without the aruco module", allow_module_level=True)
if not (Path(CALIB_REPO) / "src" / "yor_v3_calib").is_dir():
    pytest.skip(f"yor-v3-calib checkout not found at {CALIB_REPO}", allow_module_level=True)

import calib_tools as ct  # noqa: E402

C, L = ct._import_calib(CALIB_REPO)

K = np.array([[1110.2, 0.0, 1279.5], [0.0, 1110.2, 959.5], [0.0, 0.0, 1.0]])
W, H = 2560, 1920
T_C_B = np.array([[-0.999849, -0.003145, -0.017089, 0.069407],
                  [0.013357, 0.489944, -0.871651, 0.558646],
                  [0.011114, -0.871748, -0.489828, 0.236901],
                  [0.0, 0.0, 0.0, 1.0]])
HOME = np.array([0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0])
SIZE_M = 0.065


def _project(xyz):
    p = cv2.projectPoints(xyz.reshape(-1, 3), cv2.Rodrigues(T_C_B[:3, :3])[0], T_C_B[:3, 3], K, np.zeros(5))[0]
    return p.reshape(-1, 4, 2)


def _render(uv):
    """Gray image with the id-0 marker (white rim) warped to the four corners."""
    edge, pad = 168, 40
    tmpl = np.full((edge + 2 * pad, edge + 2 * pad), 255, np.uint8)
    tmpl[pad:pad + edge, pad:pad + edge] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), ct.ARUCO_ID, edge)
    src = np.array([[pad, pad], [pad + edge, pad], [pad + edge, pad + edge], [pad, pad + edge]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    canvas = np.full((H, W), 120, np.uint8)
    cv2.warpPerspective(tmpl, Hm, (W, H), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def make_dataset(root: Path, n=16, seed=0, wrist=0.0):
    rng = np.random.default_rng(seed)
    q = HOME + rng.uniform(-0.1, 0.1, size=(4 * n, 7))
    q[:, 6] += wrist
    uv_all = _project(C.marker_points(q, "left", marker_size_m=SIZE_M))
    # the real camera looks down at the mount, so the marker sits low in the
    # frame; keep only the poses whose marker is fully inside the image
    inside = ((uv_all[..., 0] > 60) & (uv_all[..., 0] < W - 60)
              & (uv_all[..., 1] > 60) & (uv_all[..., 1] < H - 60)).all(axis=1)
    assert inside.sum() >= n, f"only {inside.sum()} of {4 * n} synthetic poses land inside the image"
    q, uv_true = q[inside][:n], uv_all[inside][:n]
    root.mkdir(parents=True)
    log, plain = [], []
    for k in range(n):
        cand = 2 * k + 1                                   # gaps like the real collector
        img = _render(uv_true[k])
        cv2.imwrite(str(root / f"candidate_{cand:03d}.png"), img)
        p = L.aruco_corners_for_id(img, ct.ARUCO_ID, refine_template=False)
        assert p is not None
        plain.append(p)
        log.append({"candidate": cand, "sample": k + 1, "marker_found": True})
        log.append({"candidate": cand + 1, "sample": None, "marker_found": False})
    np.save(root / "joint_angles_left.npy", q)
    np.save(root / "two_d_coordinates_left.npy", np.asarray(plain))
    np.savez(root / "camera_intrinsics.npz", camera_matrix=K, dist_coeffs=np.zeros(5))
    (root / "collection_log.json").write_text(json.dumps(log))
    (root / "settings.json").write_text(json.dumps({"arm": "left", "refine_template": False}))
    return q, uv_true


@pytest.fixture(scope="module")
def two_sets(tmp_path_factory):
    base = tmp_path_factory.mktemp("synthetic")
    a = base / "setA"
    b = base / "setB"
    qa, uva = make_dataset(a, seed=1)
    qb, uvb = make_dataset(b, seed=2, wrist=0.6)
    return base, (a, qa, uva), (b, qb, uvb)


def run(*argv):
    ct.main(["--calib-repo", CALIB_REPO, *map(str, argv)])


def test_refine_matches_projected_corners(two_sets, capsys):
    base, (a, qa, uva), _ = two_sets
    run("refine", a, "--out", base / "setA_refined")
    out = capsys.readouterr().out
    assert "16/16 samples refined" in out
    assert "max 0.00 px" in out                 # stored arrays and images are paired
    ref = np.load(base / "setA_refined" / "two_d_coordinates_left.npy")
    err = np.linalg.norm(ref - uva, axis=2).mean(axis=1)
    assert np.median(err) < 1.0 and err.max() < 2.0
    np.testing.assert_array_equal(np.load(base / "setA_refined" / "joint_angles_left.npy"), qa)
    settings = json.loads((base / "setA_refined" / "settings.json").read_text())
    assert settings["refine_template"] is True and settings["aruco_id"] == ct.ARUCO_ID
    with pytest.raises(SystemExit):
        run("refine", a, "--out", base / "setA_refined")      # refuses to overwrite


def test_combine_fit_loo_and_sweep(two_sets, capsys):
    base, (a, *_), (b, *_) = two_sets
    if not (base / "setA_refined").exists():
        run("refine", a, "--out", base / "setA_refined")
    run("refine", b, "--out", base / "setB_refined")
    run("combine", base / "both", base / "setA_refined", base / "setB_refined")
    src = np.load(base / "both" / "source_set.npy")
    assert len(src) == 32 and set(src) == {"setA_refined", "setB_refined"}
    # the package's own fitter recovers the true camera pose
    ds = ct.load_dataset(base / "both")
    T, e, m = C.estimate_extrinsics(C.marker_points(ds["q"], "left", marker_size_m=SIZE_M), ds["uv"], K, np.zeros(5))
    assert m.sum() == 32 and e.max() < 1.5
    assert np.linalg.norm(ct.camera_centre(T) - ct.camera_centre(T_C_B)) < 0.005
    capsys.readouterr()
    run("loo", base / "both", "--no-mount")
    out = capsys.readouterr().out
    assert "held out setA_refined" in out and "held out setB_refined" in out
    assert "FAILED" not in out
    run("sweep-size", base / "both", "--sizes", 65)
    out = capsys.readouterr().out
    assert "32/32" in out


def test_show_compare_and_crops(two_sets, tmp_path, capsys):
    base, (a, qa, uva), _ = two_sets
    e = np.zeros(len(qa))
    C.save_calibration(tmp_path / "cal_a.npz", T_C_B, K, np.zeros(5), side="left", errors=e,
                       inliers=np.ones(len(qa), bool), marker_size_m=SIZE_M)
    shifted = T_C_B.copy()
    shifted[:3, 3] += T_C_B[:3, :3] @ np.array([0.0, 0.0, -0.010])    # camera 10 mm further along +z of B
    C.save_calibration(tmp_path / "cal_b.npz", shifted, K, np.zeros(5), side="left", errors=e,
                       inliers=np.ones(len(qa), bool), marker_size_m=SIZE_M)
    run("show", tmp_path / "cal_a.npz")
    out = capsys.readouterr().out
    assert "T_C_B" in out and "T_B_C" in out and "+0.069407" in out
    run("compare", tmp_path / "cal_a.npz", tmp_path / "cal_b.npz", "--data", a)
    out = capsys.readouterr().out
    assert "|10.0 mm|" in out and "relative rotation: 0.00 deg" in out
    run("compare", tmp_path / "cal_a.npz", "--b-json", json.dumps(T_C_B.tolist()))
    out = capsys.readouterr().out
    assert "|0.0 mm|" in out
    run("crops", a, "--out", tmp_path / "m.png", "--samples", 1, 8, 16)
    assert (tmp_path / "m.png").exists()
    assert cv2.imread(str(tmp_path / "m.png")).shape[1] == 3 * 400
