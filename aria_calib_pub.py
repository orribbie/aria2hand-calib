#!/usr/bin/env python3
"""Publish rectified Aria RGB for yor-v3-calib, from glasses on Thor's USB.

yor-v3-calib's client (`src/yor_v3_calib/live.py`) subscribes to two commlink
topics on one port and treats the camera as distortion-free:

    online_calib  {'K_rgb': 3x3, 'rgb_w': int, 'rgb_h': int}
    images        {'rgb': JPEG bytes}     # must decode to exactly rgb_w x rgb_h

Nothing on Thor produced those. aria2robot's publisher is hands-only by design
("RGB / stereo streams + stereo rectifier are gone", stream_pub.py:11) and its
teleop profile carries no RGB stream, so this fills the gap.

The Aria RGB camera is a fisheye, and the client assumes zero distortion, so
frames are rectified here into a linear (pinhole) model with
projectaria_tools, and the K published is that model's -- the two must agree or
the extrinsics are silently wrong.

Runs in the `aria2robot` env (Aria SDK + projectaria_tools + commlink). JPEG is
encoded with Pillow rather than OpenCV so the env needs no new packages.

    conda activate aria2robot
    python aria_calib_pub.py                       # USB, profile9, port 5555
    python aria_calib_pub.py       --check         # print K + frame size, exit

Then, in another terminal:

    cd ~/yor-v3-calib
    ./.venv/bin/yor-calib-collect --arm left --aruco-id 0 \\
        --camera-host 127.0.0.1 --robot-host 100.98.224.40 --data data/session01

If no frames arrive, the streaming profile has no RGB stream: try another
--profile, or pass --profile-json with a profile that enables rgb.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time

import numpy as np


def build_rectifier(device_calib, width, height, focal=None):
    """Fisheye 'camera-rgb' -> linear model of the same size. Returns (fn, K)."""
    from projectaria_tools.core import calibration as C

    src = device_calib.get_camera_calib("camera-rgb")
    if src is None:
        raise RuntimeError("device calibration has no 'camera-rgb'")
    f = float(focal) if focal else float(np.mean(src.get_focal_lengths()))
    dst = C.get_linear_camera_calibration(int(width), int(height), f, "camera-rgb",
                                          src.get_transform_device_camera())
    fx, fy = dst.get_focal_lengths()
    cx, cy = dst.get_principal_point()
    K = np.array([[float(fx), 0.0, float(cx)],
                  [0.0, float(fy), float(cy)],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    def rectify(img: np.ndarray) -> np.ndarray:
        return np.asarray(C.distort_by_calibration(img, dst, src))

    return rectify, K


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=5555, help="commlink publish port")
    ap.add_argument("--bind", default="*")
    ap.add_argument("--interface", default="usb", choices=("usb", "wifi_sta", "wifi_sap"))
    ap.add_argument("--profile", default="profile9",
                    help="built-in streaming profile; it must include an RGB stream")
    ap.add_argument("--profile-json", default=None,
                    help="path to a custom profile JSON (takes priority over --profile)")
    ap.add_argument("--focal", type=float, default=None,
                    help="focal length for the rectified pinhole model (default: the "
                         "fisheye's mean focal). Lower it to keep more field of view.")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality")
    ap.add_argument("--rate", type=float, default=10.0, help="max publish rate, Hz")
    ap.add_argument("--check", action="store_true",
                    help="wait for one frame, print K and size, then exit")
    args = ap.parse_args()
    rc = [0]          # exit status, read by the teardown below

    import aria.sdk_gen2 as sdk_gen2
    import aria.stream_receiver as receiver
    from commlink import Publisher
    from PIL import Image

    iface = {"usb": sdk_gen2.StreamingInterface.USB_NCM,
             "wifi_sta": sdk_gen2.StreamingInterface.WIFI_STA,
             "wifi_sap": sdk_gen2.StreamingInterface.WIFI_SAP}[args.interface]

    print(f"[aria-pub] connecting over {args.interface}…", flush=True)
    client = sdk_gen2.DeviceClient()
    device = client.connect()

    cfg = sdk_gen2.HttpStreamingConfig()
    if args.profile_json:
        cfg.profile_json = args.profile_json
    else:
        cfg.profile_name = args.profile
    cfg.streaming_interface = iface
    device.set_streaming_config(cfg)

    state: dict = {"rectify": None, "K": None, "size": None, "frames": 0,
                   "published": 0, "calib": None}
    lock = threading.Lock()
    # Set before teardown: an exception crossing back into the SDK's C++
    # callback thread aborts the process ("FATAL: exception not rethrown"),
    # which is what a core dump on exit was.
    stopping = threading.Event()
    pub = Publisher(args.bind, port=args.port)
    last_pub = [0.0]
    period = 1.0 / max(0.1, args.rate)

    def calib_cb(device_calib) -> None:
        if stopping.is_set():
            return
        with lock:
            state["calib"] = device_calib
        print("[aria-pub] factory calibration received", flush=True)

    def _rgb_impl(img, image_record, received_at) -> None:
        with lock:
            state["frames"] += 1
            if state["rectify"] is None:
                if state["calib"] is None:
                    return                      # calibration not in yet
                h, w = img.shape[:2]
                try:
                    state["rectify"], state["K"] = build_rectifier(
                        state["calib"], w, h, args.focal)
                    state["size"] = (w, h)
                    print(f"[aria-pub] rgb {w}x{h}, rectified K: "
                          f"fx={state['K'][0,0]:.1f} fy={state['K'][1,1]:.1f} "
                          f"cx={state['K'][0,2]:.1f} cy={state['K'][1,2]:.1f}", flush=True)
                except Exception as exc:
                    print(f"[aria-pub] rectifier failed: {exc}", flush=True)
                    return
            rectify, K, size = state["rectify"], state["K"], state["size"]

        now = time.monotonic()
        if now - last_pub[0] < period:
            return
        last_pub[0] = now
        try:
            rect = rectify(img)
            if rect.ndim == 2:
                rect = np.stack([rect] * 3, axis=-1)
            # live.py decodes with cv2.IMREAD_COLOR (BGR) and only checks the
            # size, so channel order affects nothing but the preview; ArUco
            # detection runs on gray.
            buf = io.BytesIO()
            Image.fromarray(rect[:, :, :3].astype(np.uint8)).save(
                buf, format="JPEG", quality=args.quality)
            pub.publish("online_calib", {"K_rgb": K, "rgb_w": int(size[0]),
                                         "rgb_h": int(size[1])})
            sync = sr.get_time_sync()
            latency_ms = (float(sync.compute_latency_ms(image_record.capture_timestamp_ns))
                          if sync is not None and sync.is_valid() else None)
            published_at = time.time()
            pub.publish("images", {"rgb": buf.getvalue(),
                                   "capture_timestamp_ns": int(image_record.capture_timestamp_ns),
                                   "received_at": received_at,
                                   "published_at": published_at,
                                   "capture_time_unix": (published_at - latency_ms / 1000
                                                         if latency_ms is not None else None),
                                   "latency_ms": latency_ms})
            with lock:
                state["published"] += 1
        except Exception as exc:
            print(f"[aria-pub] publish failed: {exc}", flush=True)

    # Keep decoding callbacks short: rectification must not back up the video stream.
    pending = [None]
    ready = threading.Event()
    pending_lock = threading.Lock()

    def publish_worker():
        while not stopping.is_set():
            if not ready.wait(.2):
                continue
            with pending_lock:
                item = pending[0]
                pending[0] = None
                ready.clear()
            if item is not None:
                try:
                    _rgb_impl(*item)
                except Exception as exc:
                    print(f"[aria-pub] worker error: {exc}", flush=True)

    def rgb_cb(image_data, image_record) -> None:
        """Nothing may propagate out of here -- see `stopping` above."""
        if stopping.is_set():
            return
        try:
            item = (image_data.to_numpy_array().copy(), image_record, time.time())
            with pending_lock:
                pending[0] = item
                ready.set()
        except Exception as exc:                      # noqa: BLE001 - deliberate
            print(f"[aria-pub] rgb callback error: {exc}", flush=True)

    sr = receiver.StreamReceiver(enable_image_decoding=True)
    sc = sdk_gen2.HttpServerConfig() if hasattr(sdk_gen2, "HttpServerConfig") else None
    if sc is not None:
        sr.set_server_config(sc)
    sr.register_device_calib_callback(calib_cb)
    sr.register_rgb_callback(rgb_cb)
    threading.Thread(target=publish_worker, daemon=True).start()

    print("[aria-pub] starting streaming…", flush=True)
    device.start_streaming()
    threading.Thread(target=sr.start_server, daemon=True).start()

    t0 = time.monotonic()
    try:
        while True:
            time.sleep(1.0)
            with lock:
                f, p, size = state["frames"], state["published"], state["size"]
            el = time.monotonic() - t0
            if args.check:
                if size is not None:
                    print(f"[aria-pub] OK — {size[0]}x{size[1]}, {f} frames in {el:.0f}s")
                    return 0
                if el > 20:
                    print("[aria-pub] no RGB frames in 20 s. The streaming profile "
                          "probably has no RGB stream — try --profile profile12/18, "
                          "or --profile-json with rgb enabled.")
                    rc[0] = 1
                    return 1
            else:
                print(f"[aria-pub] {el:5.0f}s | rgb frames {f} | published {p}"
                      + ("" if size else "  (waiting for frames + calibration)"),
                      flush=True)
                if f == 0 and el > 20:
                    print("[aria-pub] still no RGB — see --profile note above.", flush=True)
    except KeyboardInterrupt:
        print("\n[aria-pub] stopping")
    finally:
        # Order matters: quiesce the callbacks, let in-flight ones drain, stop the
        # receiver's server thread, then the device, then the connection, and only
        # last the socket the callbacks publish through.
        stopping.set()
        time.sleep(0.25)
        for label, fn in (("stop_server", getattr(sr, "stop_server", None)),
                          ("stop_streaming", getattr(device, "stop_streaming", None)),
                          # DeviceClient.disconnect takes the device it handed back.
                          ("disconnect", (lambda: client.disconnect(device)))):
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:
                print(f"[aria-pub] {label} failed: {exc}", flush=True)
        time.sleep(0.25)
        try:
            pub.stop()
        except Exception:
            pass
        # The SDK's native objects are destroyed by the interpreter after this
        # returns, and that raced into std::terminate ("terminate called without
        # an active exception", core dumped) even with the callbacks quiesced.
        # Ours are all flushed by now, so leave without running those destructors.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
