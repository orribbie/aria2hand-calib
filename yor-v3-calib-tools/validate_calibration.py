"""Evaluate a saved calibration on a separate collected dataset, without fitting."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from yor_v3_calib.calibration import marker_points
from yor_v3_calib.config import load_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    args = parser.parse_args()
    T = load_calibration(args.calibration)
    with np.load(args.calibration) as f:
        K, D = f['camera_matrix'], f['dist_coeffs']
        side = str(f['arm'])
        size = float(f['marker_size_m']) if 'marker_size_m' in f else .065
        correction = f['marker_correction'] if 'marker_correction' in f else None
    with np.load(args.data / 'camera_intrinsics.npz') as f:
        if not np.allclose(K, f['camera_matrix']) or not np.allclose(D, f['dist_coeffs']):
            raise ValueError('Validation intrinsics differ from calibration')
    xyz = marker_points(np.load(args.data / f'joint_angles_{side}.npy'), side,
                        marker_size_m=size, marker_correction=correction)
    uv = np.load(args.data / f'two_d_coordinates_{side}.npy')
    predicted = cv2.projectPoints(xyz.reshape(-1, 3), cv2.Rodrigues(T[:3, :3])[0],
                                 T[:3, 3], K, D)[0].reshape(uv.shape)
    errors = np.linalg.norm(predicted - uv, axis=2)
    pose_errors = errors.mean(axis=1)
    summary = dict(calibration=str(args.calibration.resolve()), samples=len(uv),
                   marker_size_mm=size*1000, mean_corner_error_px=float(errors.mean()),
                   median_pose_error_px=float(np.median(pose_errors)),
                   max_pose_error_px=float(pose_errors.max()),
                   max_corner_error_px=float(errors.max()),
                   poses_below_3px=int((pose_errors < 3).sum()),
                   pose_errors_px=pose_errors.tolist(), corner_errors_px=errors.tolist())
    out = args.data / 'validation'
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(summary, indent=2))
    log_path = args.data / 'collection_log.json'
    if log_path.exists():
        log = {e['sample']-1: e for e in json.loads(log_path.read_text()) if 'sample' in e}
        for i in range(len(uv)):
            entry = log[i]
            frame = cv2.imread(str(args.data / f"candidate_{entry['candidate']:03d}.png"))
            if frame is None:
                continue
            for observed, projected in zip(uv[i], predicted[i]):
                cv2.circle(frame, tuple(np.rint(observed).astype(int)), 4, (0, 255, 0), 1)
                cv2.drawMarker(frame, tuple(np.rint(projected).astype(int)), (0, 0, 255),
                               cv2.MARKER_CROSS, 12, 1)
            points = np.concatenate([uv[i], predicted[i]])
            lo = np.maximum(0, np.floor(points.min(axis=0)-30).astype(int))
            hi = np.minimum([frame.shape[1], frame.shape[0]],
                            np.ceil(points.max(axis=0)+30).astype(int))
            crop = frame[lo[1]:hi[1], lo[0]:hi[0]]
            cv2.imwrite(str(out / f'pose_{i+1:03d}.png'), crop)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
