"""Synthetic wrist emitter — validates the stage-2 wrist receiver without WiLoR.
Sends a wrist pose as 12 float64 (3x4 [R|t], the sim_teleop VIVE wire format)
over UDP 5012, rotating slowly + small translation, the same format the real
WiLoR emitter will send (r0w / pred_cam_t_full packed as [R|t]).

    term A:  python -m shadow_ext.teleop_driver pick_bucket --view --recv-wrist
    term B:  python -m shadow_ext.wrist_emitter_test

The Panda flange should rotate (and drift slightly) tracking the streamed pose.
"""
from __future__ import annotations
import socket
import time
import numpy as np
import mujoco

from .teleop_driver import _WRIST_UDP_PORT


def _pose_at(t: float) -> np.ndarray:
    """3x4 [R|t]: oscillating tilt about a rotating axis + small translation."""
    angle = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    axis = np.array([np.cos(2 * np.pi * 0.1 * t), np.sin(2 * np.pi * 0.1 * t), 0.0])
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    pose = np.zeros((3, 4))
    pose[:3, :3] = m.reshape(3, 3)
    pose[:3, 3] = np.array([0.05 * np.sin(2 * np.pi * 0.15 * t), 0.0, 0.0])
    return pose


def main(host: str = "127.0.0.1", port: int = _WRIST_UDP_PORT, hz: float = 50.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / hz
    print(f"emitting wrist 3x4 -> {host}:{port} at {hz:.0f}Hz (tilt + small translate)")
    t0 = time.time()
    try:
        while True:
            pose = _pose_at(time.time() - t0)
            sock.sendto(pose.astype(np.float64).tobytes(), (host, port))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
