"""Stream selected MuJoCo cameras to video without retaining RGB frames."""
from pathlib import Path
import mujoco

class Recorder:
    _SHOT_CAMERAS = ('front', 'left', 'right')

    def __init__(self, model, cam='front', width=1280, height=720, fps=30,
                 every=None, video_path=None):
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('Video dimensions and fps must be positive')
        self._model, self._cam = model, cam.split(',')[0]
        self._video_cams = list(dict.fromkeys(cam.split(',')))
        missing = set(self._video_cams) - set(self.camera_names())
        if missing:
            raise ValueError(f'Unknown cameras: {sorted(missing)}; available: {self.camera_names()}')
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._fps = float(fps) if every is None else 1 / (model.opt.timestep * every)
        self._writers, self._paths = {}, []
        self._frame_index, self._origin, self._last = 0, None, None
        if video_path:
            try:
                self.start_video(video_path)
            except BaseException:
                self.close()
                raise

    def start_video(self, path):
        import imageio.v2 as imageio
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        paths = {c: p.with_name(f'{p.stem}_{c}{p.suffix}') for c in self._video_cams}
        if any(p.exists() for p in paths.values()):
            raise FileExistsError('Video already exists; choose a new output name')
        for cam, output in paths.items():
            self._writers[cam] = imageio.get_writer(str(output), fps=self._fps,
                                                   codec='libx264', quality=8, macro_block_size=2)
            self._paths.append(str(output))

    def _render(self, data, cam=None):
        self._renderer.update_scene(data, camera=cam or self._cam)
        return self._renderer.render().copy()

    def capture(self, data):
        frames = {c: self._render(data, c) for c in self._video_cams}
        for cam, frame in frames.items():
            self._writers[cam].append_data(frame)
        self._last = frames
        self._frame_index += 1
        return frames

    def maybe_capture(self, data, step=0, elapsed=None):
        stamp = float(data.time) if elapsed is None else float(elapsed)
        if self._origin is None:
            self._origin = stamp
        wanted = int((stamp - self._origin) * self._fps + 1e-8)
        if wanted < self._frame_index:
            return
        while self._last is not None and self._frame_index < wanted:
            for cam, frame in self._last.items():
                self._writers[cam].append_data(frame)
            self._frame_index += 1
        self.capture(data)

    def shot(self, data, path):
        import imageio.v3 as iio
        iio.imwrite(path, self._render(data))

    def camera_names(self):
        return [self._model.camera(i).name or f'cam{i}' for i in range(self._model.ncam)]

    def shot_all(self, data, prefix):
        import imageio.v3 as iio
        paths = []
        for cam in self._SHOT_CAMERAS:
            if cam in self.camera_names():
                path = f'{prefix}_{cam}.png'
                iio.imwrite(path, self._render(data, cam))
                paths.append(path)
        return paths

    def save_video(self, path=None):
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        return self._paths

    def close(self):
        self.save_video()
        self._renderer.close()
