"""Offscreen recording for thesis figures / videos. No RL wiring.

Wraps mujoco.Renderer (offscreen RGB) over any camera already defined in the
scene (front, back, left, right, top0, handcam_rgb). Two uses, same renderer:

  - photo: one RGB frame -> PNG (e.g. at the grasp moment).
  - video: a frame every K steps -> MP4 via imageio.

Independent of the env's RL observation cameras. Pure rendering.
"""
from __future__ import annotations
import numpy as np
import mujoco


class Recorder:
    #: Subset of scene cameras actually wanted for thesis-figure stills
    #: (2026-08-26: user asked to drop top0/handcam_rgb, front/left/right
    #: already cover the operator-comparable angles). --record now captures
    #: ALL of these simultaneously to separate videos (2026-09-07: matches
    #: the original 3-angle-per-shot figure format, but hands-free/continuous
    #: instead of needing a keypress per instant -- both hands are busy
    #: driving the real hand + arm live, there's no free hand for --shot).
    _SHOT_CAMERAS = ("front", "left", "right")

    def __init__(self, model, cam: str = "front", width: int = 1280,
                 height: int = 720, fps: int = 30, every: int = 15):
        # 1920x1080 x 3 simultaneous cameras measured ~55x slower than real
        # time on this machine's integrated GPU (4s task -> 3:40 wall clock,
        # confirmed live 2026-09-07) -- unusable during a live grasp attempt.
        # 1280x720 + capturing every 15th step (not 5th) cuts rendered-pixel
        # work by roughly 6-7x; re-measure if this machine's GPU changes.
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._model = model
        self._cam = cam
        self._fps = fps
        self._every = max(1, int(every))
        self._frames: list[np.ndarray] = []  # single-camera path (self._cam), kept for shot()
        self._video_cams = [c for c in self._SHOT_CAMERAS if c in set(self.camera_names())]
        self._video_frames: dict[str, list[np.ndarray]] = {c: [] for c in self._video_cams}

    def _render(self, data, cam: str | None = None) -> np.ndarray:
        self._renderer.update_scene(data, camera=cam or self._cam)
        return self._renderer.render()

    def maybe_capture(self, data, step: int) -> None:
        """Call each sim step; grabs a frame every `every` steps, from ALL
        _SHOT_CAMERAS at once, for save_video()."""
        if step % self._every == 0:
            for cam in self._video_cams:
                self._video_frames[cam].append(self._render(data, cam))

    def shot(self, data, path: str) -> None:
        """Save a single PNG now."""
        import imageio.v3 as iio
        iio.imwrite(path, self._render(data))

    def camera_names(self) -> list[str]:
        return [self._model.camera(i).name or f"cam{i}" for i in range(self._model.ncam)]

    def shot_all(self, data, path_prefix: str) -> list[str]:
        """Save one PNG per camera in _SHOT_CAMERAS (skips any not in the model)."""
        import imageio.v3 as iio
        available = set(self.camera_names())
        paths = []
        for name in self._SHOT_CAMERAS:
            if name not in available:
                continue
            self._renderer.update_scene(data, camera=name)
            p = f"{path_prefix}_{name}.png"
            iio.imwrite(p, self._renderer.render())
            paths.append(p)
        return paths

    def save_video(self, path: str) -> list[str]:
        """Write one video per camera in _SHOT_CAMERAS, path suffixed with
        the camera name (e.g. run.mp4 -> run_front.mp4, run_left.mp4,
        run_right.mp4). Returns the list of paths actually written."""
        import os
        import imageio
        # Real playback-rate fps: physics runs at 1/timestep steps/sec, we
        # keep 1-in-`every` of those -- NOT self._fps (a target that was
        # never actually the physics rate, e.g. 30 vs a real ~500Hz sim at
        # timestep=0.002). The old formula (self._fps // every) played back
        # ~16x too slow (confirmed live 2026-09-07, visibly slow-motion).
        sim_hz = 1.0 / float(self._model.opt.timestep)
        out_fps = max(1, round(sim_hz / self._every))
        stem, ext = os.path.splitext(path)
        written = []
        for cam, frames in self._video_frames.items():
            if not frames:
                continue
            p = f"{stem}_{cam}{ext}"
            imageio.mimwrite(p, frames, fps=out_fps, codec="libx264", quality=10)
            written.append(p)
        return written

    def close(self) -> None:
        self._renderer.close()
