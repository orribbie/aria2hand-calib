"""Plan and collect broad camera-facing marker poses with checked joint paths.

Planning is read-only. --execute PLAN runs only a previously saved plan.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from yor_v3_calib.calibration import correct_marker_points
from yor_v3_calib.config import ARUCO_MJCF, ARM_JOINTS, CORNER_SITES, initial_q
from yor_v3_calib.live import CalibrationDataCollector, aruco_corners_for_id


class Geometry:
    def __init__(self, other_q, calibration):
        self.model = mujoco.MjModel.from_xml_path(str(ARUCO_MJCF))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = initial_q(self.model)
        for name, value in zip(ARM_JOINTS['right'], other_q):
            self.data.joint(name).qpos[0] = value
        self.joints = np.array([self.model.joint(n).id for n in ARM_JOINTS['left']])
        self.adr = self.model.jnt_qposadr[self.joints]
        self.limits = self.model.jnt_range[self.joints].copy()
        self.sites = [self.model.site(n).id for n in CORNER_SITES['left']]
        with np.load(calibration) as f:
            self.T = f['T_C_B']
            self.K = f['camera_matrix']
            self.size = float(f['marker_size_m'])
            self.correction = f['marker_correction'] if 'marker_correction' in f else np.eye(4)

    def fk(self, q):
        self.data.qpos[self.adr] = q
        mujoco.mj_forward(self.model, self.data)
        xyz = self.data.site_xpos[self.sites].copy()
        c = xyz.mean(axis=0)
        xyz = c + (xyz-c)*self.size/.065
        xyz = correct_marker_points(xyz[None], self.correction)[0]
        xyz = xyz @ self.T[:3, :3].T + self.T[:3, 3]
        a, b = (xyz[1]-xyz[0])/self.size, (xyz[3]-xyz[0])/self.size
        return xyz.mean(axis=0), np.stack([a, b, np.cross(a, b)], axis=1), xyz

    def valid(self, q, *, margin=40, max_angle=60):
        if np.any(q < self.limits[:, 0]+.015) or np.any(q > self.limits[:, 1]-.015):
            return False
        c, R, xyz = self.fk(q)
        if self.data.ncon or np.any(xyz[:, 2] < .20):
            return False
        if np.dot(R[:, 2], c/np.linalg.norm(c)) < np.cos(np.deg2rad(max_angle)):
            return False
        uv = xyz @ self.K.T
        uv = uv[:, :2]/uv[:, 2:]
        return bool(uv[:, 0].min() > margin and uv[:, 0].max() < 2560-margin
                    and uv[:, 1].min() > margin and uv[:, 1].max() < 1920-margin
                    and np.min(np.linalg.norm(np.roll(uv, -1, axis=0)-uv, axis=1)) > 35)

    def path(self, a, b):
        steps = max(2, int(np.ceil(np.max(np.abs(b-a))/.01))+1)
        return all(self.valid(a+(b-a)*s) for s in np.linspace(0, 1, steps))

    def ik(self, center, R, start):
        def residual(q):
            c, orientation, _ = self.fk(q)
            rot = cv2.Rodrigues(R.T @ orientation)[0].ravel()
            return np.r_[(c-center), .10*rot, .0005*(q-start)]
        upper = self.limits[:, 1]-.02
        # Live joint 2 stopped at 1.619 rad despite a 1.675 rad target.
        # Keep subsequent requested poses below that observed operating limit.
        upper[1] = min(upper[1], 1.60)
        lower = self.limits[:, 0]+.02
        result = least_squares(residual, np.clip(start, lower+1e-5, upper-1e-5),
                               bounds=(lower, upper),
                               max_nfev=120, diff_step=1e-4)
        c, orientation, _ = self.fk(result.x)
        angle = np.linalg.norm(cv2.Rodrigues(R.T @ orientation)[0])
        if np.linalg.norm(c-center) > .003 or angle > np.deg2rad(3):
            return None
        return result.x if self.valid(result.x, margin=70, max_angle=30) else None


def facing(center, tilt):
    z = center/np.linalg.norm(center)
    x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1) @ cv2.Rodrigues(tilt)[0]


def client():
    return CalibrationDataCollector('100.98.224.40', 5557, '127.0.0.1', 5555, 0)


def plan(args):
    c = client()
    q0 = np.asarray(c.robot.get_left_joint_positions())
    qo = np.asarray(c.robot.get_right_joint_positions())
    K = c.fetch_intrinsics()
    frame = c.get_rgb_frame(after_time=time.time())
    g = Geometry(qo, args.calibration)
    if frame.shape[:2] != (1920, 2560) or not np.allclose(K, g.K):
        raise RuntimeError('Camera geometry differs from planning calibration')
    if aruco_corners_for_id(frame, 0) is None or not g.valid(q0):
        raise RuntimeError('Initial marker must be visible and model pose contact-free')
    center, _, _ = g.fk(q0)
    base = center + [.06, -.13, .04]
    rng = np.random.default_rng(args.seed)
    goals, centers, angles = [], [], []
    current = q0.copy()
    for i in range(args.count*20):
        goal = base + rng.uniform([- .12, -.08, -.07], [.12, .08, .07])
        tilt = rng.uniform(-1, 1, 3)*np.deg2rad([12, 12, 15])
        q = g.ik(goal, facing(goal, tilt), current)
        if q is None or not g.path(current, q):
            continue
        actual, R, _ = g.fk(q)
        if centers and np.min(np.linalg.norm(np.array(centers)-actual, axis=1)) < .025:
            continue
        goals.append(q)
        centers.append(actual)
        angles.append(float(np.rad2deg(np.arccos(np.clip(np.dot(R[:, 2], actual/np.linalg.norm(actual)), -1, 1)))))
        current = q
        print(f'Planned {len(goals)}/{args.count}; center {actual.round(3)}, facing angle {angles[-1]:.1f} deg', flush=True)
        if len(goals) == args.count:
            break
    if len(goals) < args.count:
        raise RuntimeError(f'Only {len(goals)} reachable visible goals; no robot commands sent')
    # Return follows the exact checked path in reverse.
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out/'plan.npz', q0=q0, other_q=qo, goals=goals, centers=centers,
             calibration=str(args.calibration.resolve()), facing_angles_deg=angles)
    cv2.imwrite(str(out/'initial_rgb.png'), frame)
    summary = dict(count=len(goals), seed=args.seed,
                   camera_center_span_m=np.ptp(centers, axis=0).tolist(),
                   joint_span_rad=np.ptp(np.array(goals), axis=0).tolist(),
                   max_facing_angle_deg=max(angles))
    (out/'plan_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def execute(args):
    with np.load(args.execute) as f:
        q0, qo, goals = f['q0'], f['other_q'], f['goals']
        calibration = Path(str(f['calibration']))
    out = args.execute.parent
    if (out/'joint_angles_left.npy').exists() and not args.resume:
        raise RuntimeError('Refusing to overwrite an existing collection')
    g = Geometry(qo, calibration)
    c = client()
    K = c.fetch_intrinsics()
    if not np.allclose(K, g.K):
        raise RuntimeError('Live intrinsics changed since planning')
    np.savez(out/'camera_intrinsics.npz', camera_matrix=K, dist_coeffs=np.zeros(5))
    def read():
        q = np.asarray(c.robot.get_left_joint_positions(), dtype=float)
        other = np.asarray(c.robot.get_right_joint_positions(), dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all() or not g.valid(q):
            raise RuntimeError('Measured arm left the checked visible workspace')
        if other.shape != (7,) or np.max(np.abs(other-qo)) > .04:
            raise RuntimeError('Opposite arm moved')
        return q
    log = json.loads((out/'collection_log.json').read_text()) if args.resume else []
    start_index = len(log)
    if args.resume and start_index >= len(goals):
        raise RuntimeError('All planned candidates already processed')
    expected = goals[start_index] if args.resume else q0
    if np.max(np.abs(read()-expected)) > (.08 if args.resume else .02):
        raise RuntimeError('Robot moved since plan creation; replan')
    history = [q0.copy()] + [row.copy() for row in goals[:start_index]]
    def move(target):
        start = read()
        if not g.path(start, target):
            raise RuntimeError('Interpolated path failed collision/visibility checks')
        steps = max(1, int(np.ceil(np.max(np.abs(target-start))/.003)))
        for k in range(1, steps+1):
            command = start+(target-start)*k/steps
            c.robot.set_left_joint_target(command)
            time.sleep(.04)
            if k % 10 == 0 and np.max(np.abs(read()-command)) > .06:
                raise RuntimeError('Joint tracking error')
        previous, stable = read(), 0
        for _ in range(80):
            time.sleep(.1)
            q = read()
            stable = stable+1 if np.max(np.abs(q-previous)) <= .002 else 0
            previous = q
            if stable >= 5 and np.max(np.abs(q-target)) < .025:
                return q
        raise RuntimeError('Arm did not settle')
    joints = list(np.load(out/'joint_angles_left.npy')) if args.resume else []
    corners = list(np.load(out/'two_d_coordinates_left.npy')) if args.resume else []
    if len(joints) != len(corners) or len(joints) != sum('sample' in e for e in log):
        raise RuntimeError('Incomplete collection; refusing unsafe resume')
    try:
        for i, target in enumerate(goals[start_index:], start_index+1):
            if target[1] > 1.60:
                log.append(dict(candidate=i, marker_found=False,
                                skipped='Joint 2 target exceeds observed operating range'))
                (out/'collection_log.json').write_text(json.dumps(log, indent=2))
                print(f'Skipping pose {i}: joint 2 target above 1.60 rad', flush=True)
                continue
            print(f'Moving to wide pose {i}/{len(goals)}', flush=True)
            move(target)
            history.append(target.copy())
            time.sleep(.4)
            before = read()
            settled_at = time.time()
            frames, values, qs, metas = [], [], [], []
            # Multiple fresh stationary observations suppress image and encoder noise.
            for _ in range(3):
                frame = c.get_rgb_frame(after_time=settled_at)
                settled_at = time.time()
                q = read()
                uv = aruco_corners_for_id(frame, 0, refine_template=True)
                if uv is not None:
                    frames.append(frame)
                    values.append(uv)
                    qs.append(q)
                    metas.append(c.last_frame_metadata)
            q = read()
            drift = float(np.max(np.abs(q-before)))
            entry = dict(candidate=i, marker_found=bool(values), max_capture_drift_rad=drift,
                         frames=metas)
            if not values:
                cv2.imwrite(str(out/f'candidate_{i:03d}.png'), frame)
                raise RuntimeError('Marker lost at a planned facing pose; holding for inspection')
            uv = np.mean(values, axis=0)
            spread = float(np.max(np.linalg.norm(np.asarray(values)-uv, axis=2)))
            entry['corner_spread_px'] = spread
            cv2.imwrite(str(out/f'candidate_{i:03d}.png'), frames[-1])
            if len(values) >= 2 and drift <= .004 and spread < 3.:
                joints.append(np.mean(qs, axis=0))
                corners.append(uv)
                entry['sample'] = len(joints)
                np.save(out/'joint_angles_left.npy', joints)
                np.save(out/'two_d_coordinates_left.npy', corners)
                print(f'SAVED {len(joints)}; spread {spread:.2f} px', flush=True)
            else:
                print(f'Skipped unstable observation: {len(values)} frames, drift {drift:.4f}, spread {spread:.2f}', flush=True)
            log.append(entry)
            (out/'collection_log.json').write_text(json.dumps(log, indent=2))
        print('Returning along the checked path', flush=True)
        if g.path(read(), q0):
            move(q0)
        else:
            for target in reversed(history[:-1]):
                move(target)
        np.save(out/'final_left_q.npy', read())
    except BaseException:
        try:
            c.robot.set_left_joint_target(np.asarray(c.robot.get_left_joint_positions()))
        except Exception:
            pass
        raise
    print(f'Collected {len(joints)} broad poses', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, default=Path('data/calibration_fresh60_train/hand_eye_calibration_left.npz'))
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--seed', type=int, default=501)
    p.add_argument('--output', type=Path, default=Path('data/wide60_train'))
    p.add_argument('--execute', type=Path)
    p.add_argument('--resume', action='store_true', help='Resume at an interrupted planned pose')
    args = p.parse_args()
    if args.execute:
        execute(args)
    else:
        plan(args)


if __name__ == '__main__':
    main()
