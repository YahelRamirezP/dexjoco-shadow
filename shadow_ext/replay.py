"""Render recorded simulation and webcam frames on one original timeline."""
import csv
import json
import math
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from .trace_video import Recorder
from .state_recording import TraceReader

class OperatorVideo:
    def __init__(self, directory, trace_metadata):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if not self.metadata.get('complete'):
            raise ValueError('Stop operator recording before rendering synchronized videos')
        if not self.metadata.get('boot_id') or not trace_metadata.get('boot_id'):
            raise ValueError('Missing host clock identity; cannot verify synchronization')
        if any(self.metadata.get(k) != trace_metadata.get(k) for k in ('clock', 'boot_id')):
            raise ValueError('Recordings do not share the same host clock')
        if self.metadata.get('robot') != trace_metadata.get('hand'):
            raise ValueError('Operator robot differs from the simulation hand')
        with (directory / 'frames.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.timestamps = np.array([int(r['monotonic_ns']) for r in rows], dtype=np.int64)
        if not len(rows) or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError('Missing or invalid operator timestamps')
        if [int(r['frame']) for r in rows] != list(range(len(rows))):
            raise ValueError('Operator frame index is inconsistent')
        self.capture = cv2.VideoCapture(str(directory / 'camera.avi'))
        if not self.capture.isOpened():
            raise ValueError('Cannot open operator video')
        self.index, self.frame = -1, None

    def at(self, timestamp):
        wanted = int(np.searchsorted(self.timestamps, timestamp, side='right') - 1)
        if wanted < 0 or timestamp > self.timestamps[-1]:
            return None
        while self.index < wanted:
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError('Operator video ends before its timestamp index')
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.index += 1
        return self.frame

    def close(self):
        self.capture.release()

def fit_frame(frame, width, height):
    result = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None:
        cv2.putText(result, 'No operator frame', (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        return result
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (width - nw) // 2, (height - nh) // 2
    result[y:y + nh, x:x + nw] = resized
    return result

def render_trace(directory, output, cam='front', width=1280, height=720,
                 fps=30, with_operator=False):
    if not output or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError('Replay requires an output, positive fps, and positive even video dimensions')
    cam = ','.join(dict.fromkeys(c.strip() for c in cam.split(',')))
    reader = TraceReader(directory)
    recorder = operator = None
    writers = {}
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    timeline = p.with_name(p.stem + '_timeline.csv')
    manifest = p.with_name(p.stem + '_export.json')
    if timeline.exists() or manifest.exists():
        reader.close()
        raise FileExistsError('Export already exists; choose a new output name')
    first, last = int(reader.timestamps[0]), int(reader.timestamps[-1])
    count = max(1, math.ceil((last - first) * fps / 1e9) + 1)
    written, complete = [], False
    try:
        if with_operator:
            operator = OperatorVideo(Path(directory).parent / 'operator', reader.metadata)
            if operator.timestamps[-1] < first or operator.timestamps[0] > last:
                raise ValueError('Operator and simulation recordings do not overlap')
            for name in ['operator'] + [f'paired_{c}' for c in cam.split(',')]:
                dest = p.with_name(f'{p.stem}_{name}{p.suffix}')
                if dest.exists():
                    raise FileExistsError(dest)
                writers[name] = imageio.get_writer(str(dest), fps=fps, codec='libx264',
                                                  quality=8, macro_block_size=2)
                written.append(str(dest))
        recorder = Recorder(reader.model, cam=cam, width=width, height=height,
                            fps=fps, video_path=output)
        with timeline.open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_frame', 'elapsed_seconds', 'monotonic_ns',
                             'simulation_state_index', 'operator_frame_index',
                             'simulation_time', 'task_success'])
            for k in range(count):
                timestamp = first + round(k * 1e9 / fps)
                index = reader.index_at(timestamp)
                data = reader.restore(index)
                frames = recorder.capture(data)
                op_index = -1
                if operator is not None:
                    frame = operator.at(timestamp)
                    op_index = operator.index if frame is not None else -1
                    fitted = fit_frame(frame, width, height)
                    writers['operator'].append_data(fitted)
                    for name, robot_frame in frames.items():
                        writers[f'paired_{name}'].append_data(np.concatenate((fitted, robot_frame), axis=1))
                writer.writerow([k, k / fps, timestamp, index, op_index,
                                 float(data.time), reader.success_at(index)])
                if k % max(1, int(fps * 5)) == 0:
                    print(f'[replay] {k}/{count} frames', flush=True)
        complete = True
    finally:
        if recorder is not None:
            written.extend(recorder.save_video())
            recorder.close()
        for writer in writers.values():
            writer.close()
        if operator is not None:
            operator.close()
        reader.close()
        manifest.write_text(json.dumps({
            'complete': complete, 'trace': str(Path(directory).resolve()),
            'fps': fps, 'start_ns': first, 'end_ns': last,
            'frame_count': count, 'timeline': str(timeline),
            'endpoint': 'Final state included; output may extend by at most one frame interval',
            'sampling': 'last state/frame at or before each timestamp; no dynamics rerun',
            'hand': reader.metadata.get('hand'),
            'slip_compensate': reader.metadata.get('slip_compensate'),
            'float_object': reader.metadata.get('float_object'), 'videos': written,
        }, indent=2) + '\n')
    print(f'[replay] Same-timeline videos: {written}')
    return written


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', help='SESSION/sim directory')
    parser.add_argument('--record', required=True, help='Output MP4 prefix; each camera gets a suffix')
    parser.add_argument('--cam', default='front', help='Comma-separated camera names')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--with-operator', action='store_true')
    args = parser.parse_args()
    render_trace(args.trace, args.record, args.cam, args.width, args.height,
                 args.fps, args.with_operator)
