"""Synthetic finger emitter — validates the stage-1 streaming receiver without
WiLoR or a camera. Sends a qpos[24] (Menagerie order, float64) open->close sweep
over UDP 5014, the same wire format the real retargeter emitter will use.

Use to eyeball the live-finger path:
    term A:  python -m shadow_ext.teleop_driver pick_bucket --view --recv-fingers
    term B:  python -m shadow_ext.finger_emitter_test

The hand should open and close in a loop. This is the contract test for the
receiver; the real emitter swaps this sweep for WiLoRSource -> Retargeter qpos.
"""
from __future__ import annotations
import socket
import time
import numpy as np

from .teleop_driver import _SHADOW_24_ORDER, _FINGER_UDP_PORT

# Flexion targets at full close, per joint name (abduction left at 0). Mirrors
# the driver's hand-tuned _CLOSE_JOINTS so a full sweep looks like a fist.
_CLOSE = {
    "FFJ3": 1.0, "FFJ2": 1.4, "FFJ1": 1.4,
    "MFJ3": 1.0, "MFJ2": 1.4, "MFJ1": 1.4,
    "RFJ3": 1.0, "RFJ2": 1.4, "RFJ1": 1.4,
    "LFJ3": 1.0, "LFJ2": 1.4, "LFJ1": 1.4,
    "THJ5": 0.3, "THJ4": 1.1, "THJ2": 0.5, "THJ1": 1.0,
}


def _close_vec() -> np.ndarray:
    q = np.zeros(24, dtype=np.float64)
    for i, jn in enumerate(_SHADOW_24_ORDER):
        if jn in _CLOSE:
            q[i] = _CLOSE[jn]
    return q


def main(host: str = "127.0.0.1", port: int = _FINGER_UDP_PORT,
         hz: float = 50.0, period_s: float = 3.0):
    close = _close_vec()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / hz
    print(f"emitting qpos[24] -> {host}:{port} at {hz:.0f}Hz, {period_s}s open/close cycle")
    t0 = time.time()
    try:
        while True:
            t = (time.time() - t0)
            alpha = 0.5 * (1.0 - np.cos(2 * np.pi * t / period_s))  # 0..1..0 smooth
            sock.sendto((alpha * close).tobytes(), (host, port))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
