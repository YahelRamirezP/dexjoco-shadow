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
    def __init__(self, model, cam: str = "front", width: int = 1280,
                 height: int = 720, fps: int = 30, every: int = 5):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._cam = cam
        self._fps = fps
        self._every = max(1, int(every))
        self._frames: list[np.ndarray] = []

    def _render(self, data) -> np.ndarray:
        self._renderer.update_scene(data, camera=self._cam)
        return self._renderer.render()

    def maybe_capture(self, data, step: int) -> None:
        """Call each sim step; grabs a frame every `every` steps for video."""
        if step % self._every == 0:
            self._frames.append(self._render(data))

    def shot(self, data, path: str) -> None:
        """Save a single PNG now."""
        import imageio.v3 as iio
        iio.imwrite(path, self._render(data))

    def save_video(self, path: str) -> None:
        if not self._frames:
            return
        import imageio
        out_fps = max(1, self._fps // self._every)
        imageio.mimwrite(path, self._frames, fps=out_fps,
                         codec="libx264", quality=8)

    def close(self) -> None:
        self._renderer.close()
