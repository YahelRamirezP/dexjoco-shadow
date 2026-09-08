"""Full MuJoCo state traces with a shared host clock and frozen model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from .trial import Evaluation

import mujoco
import numpy as np


def clock_identity():
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'clock': 'time.monotonic_ns',
            'boot_id': boot.read_text().strip() if boot.exists() else None}


class StateTrace:
    """Flush small independent chunks so memory stays bounded during long takes."""

    def __init__(self, directory, model, metadata=None, chunk_size=500):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model = model
        self.chunk_size = chunk_size
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._size = mujoco.mj_stateSize(model, self.spec)
        self._rows = []
        self._chunks = 0
        self._closed = False
        self._events = None
        model_path = self.directory / 'model.mjb'
        mujoco.mj_saveModel(model, str(model_path))
        self.metadata = {**(metadata or {}), **clock_identity(), 'format_version': 2,
                         'created_at_utc': datetime.now(timezone.utc).isoformat(),
                         'mujoco_version': mujoco.__version__, 'state_spec': int(self.spec),
                         'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         'timestep': float(model.opt.timestep),
                         'gravity': model.opt.gravity.tolist(),
                         'complete': False, 'frames': 0, 'chunks': 0,
                         'success_ever': False, 'first_success_sim_time': None,
                         'first_success_monotonic_ns': None}
        self.evaluation = Evaluation(self.metadata.get('evaluation_start'),
                                     self.metadata.get('time_limit_wall_s'))
        self.metadata['event_counts'] = {}
        self._write_metadata()
        self._events = (self.directory / 'events.jsonl').open('x')

    def _write_metadata(self):
        p = self.directory / 'metadata.tmp.json'
        p.write_text(json.dumps(self.metadata, indent=2) + '\n')
        p.replace(self.directory / 'metadata.json')

    def event(self, kind, data, *, timestamp_ns=None, **details):
        if self._closed:
            raise RuntimeError('State trace is closed')
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        event = {'type': kind, 'monotonic_ns': stamp, 'sim_time': float(data.time),
                 'preceding_state_index': self.metadata['frames'] - 1,
                 'evaluation_active': self.evaluation.start_ns is not None, **details}
        self._events.write(json.dumps(event, allow_nan=False) + '\n')
        self._events.flush()
        counts = self.metadata['event_counts']
        counts[kind] = counts.get(kind, 0) + 1

    def start_evaluation(self, data, timestamp_ns=None):
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        if not self.evaluation.start(stamp, data.time):
            return False
        self.event('evaluation_start', data, timestamp_ns=stamp)
        self.metadata['evaluation'] = self.evaluation.summary(stamp, data.time)
        self._write_metadata()
        return True

    def append(self, data, *, target_ctrl, slip_bias, success=False, timestamp_ns=None,
               task_metrics=None):
        if self._closed:
            raise RuntimeError('State trace is closed')
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        state = np.empty(self._size)
        mujoco.mj_getState(self.model, data, state, self.spec)
        self._rows.append((stamp, float(data.time), state, self.model.body_gravcomp.copy(),
                           np.array(target_ctrl, copy=True), np.array(slip_bias, copy=True),
                           bool(success), json.dumps(task_metrics, allow_nan=False),
                           self.evaluation.start_ns is not None))
        self.metadata['final_task_metrics'] = task_metrics
        self.evaluation.observe(success, stamp, data.time)
        self.metadata['evaluation'] = self.evaluation.summary(stamp, data.time)
        self.metadata.setdefault('start_ns', stamp)
        self.metadata.setdefault('start_sim_time', float(data.time))
        self.metadata['end_sim_time'] = float(data.time)
        if success and not self.metadata['success_ever']:
            self.metadata.update(success_ever=True, first_success_sim_time=float(data.time),
                                 first_success_monotonic_ns=stamp)
        self.metadata['end_ns'] = stamp
        self.metadata['frames'] += 1
        if len(self._rows) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        columns = list(zip(*self._rows))
        keys = ('monotonic_ns', 'sim_time', 'state', 'body_gravcomp',
                'target_ctrl', 'slip_bias', 'success', 'task_metrics', 'evaluation_active')
        path = self.directory / f'{self._chunks:06d}.npz'
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, **{k: np.asarray(v) for k, v in zip(keys, columns)})
        temporary.replace(path)
        self._rows.clear()
        self._chunks += 1
        self.metadata['chunks'] = self._chunks
        self._write_metadata()

    def close(self, reason='finished'):
        if self._closed:
            return
        self.flush()
        self.metadata.update(complete=True, stop_reason=reason)
        start = self.metadata.get('start_ns')
        first_success = self.metadata.get('first_success_monotonic_ns')
        self.metadata['time_to_success_wall_s'] = (
            (first_success - start) / 1e9 if first_success is not None else None)
        self.metadata['time_to_success_sim_s'] = (
            self.metadata['first_success_sim_time'] - self.metadata['start_sim_time']
            if first_success is not None else None)
        self.metadata['elapsed_wall_s'] = (
            (self.metadata['end_ns'] - start) / 1e9 if start is not None else 0)
        self.metadata['elapsed_sim_s'] = (
            self.metadata['end_sim_time'] - self.metadata['start_sim_time']
            if start is not None else 0)

        self._write_metadata()
        if self._events is not None:
            self._events.close()
        self._closed = True
        print(f'[state trace] {self.metadata["frames"]} states -> {self.directory}')


class TraceReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        if self.metadata.get('format_version') not in (1, 2):
            raise ValueError('Unsupported state trace format')
        if self.metadata['mujoco_version'] != mujoco.__version__:
            raise ValueError(f'Trace needs MuJoCo {self.metadata["mujoco_version"]}; '
                             f'current version is {mujoco.__version__}')
        model_path = self.directory / 'model.mjb'
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.metadata['model_sha256']:
            raise ValueError('Recorded model checksum does not match')
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.paths = sorted(self.directory.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz'))
        stamps, lengths = [], []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as chunk:
                stamps.append(chunk['monotonic_ns'])
                lengths.append(len(stamps[-1]))
        if not stamps or not sum(lengths):
            raise ValueError('No saved simulation frames')
        self.timestamps = np.concatenate(stamps)
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError('Trace timestamps are not monotonic')
        self.offsets = np.r_[0, np.cumsum(lengths)]
        self._chunk_id = None
        self._chunk = None
        self.spec = mujoco.mjtState(self.metadata['state_spec'])
        if not self.metadata['complete']:
            print('[replay] Incomplete take: replaying its saved chunks only.')

    def restore(self, index):
        index = int(index)
        if not 0 <= index < len(self.timestamps):
            raise IndexError(index)
        chunk_id = int(np.searchsorted(self.offsets[1:], index, side='right'))
        if chunk_id != self._chunk_id:
            with np.load(self.paths[chunk_id], allow_pickle=False) as chunk:
                self._chunk = {key: chunk[key] for key in chunk.files}
            self._chunk_id = chunk_id
        row = index - self.offsets[chunk_id]
        state = self._chunk['state'][row]
        self.model.body_gravcomp[:] = self._chunk['body_gravcomp'][row]
        mujoco.mj_setState(self.model, self.data, state, self.spec)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def index_at(self, timestamp_ns):
        return int(np.clip(np.searchsorted(self.timestamps, timestamp_ns, side='right') - 1,
                           0, len(self.timestamps) - 1))

    def task_metrics_at(self, index):
        self.restore(index)
        if 'task_metrics' not in self._chunk:
            return None
        row = int(index) - self.offsets[self._chunk_id]
        return json.loads(str(self._chunk['task_metrics'][row]))

    def success_at(self, index):
        self.restore(index)
        row = int(index) - self.offsets[self._chunk_id]
        return bool(self._chunk['success'][row])

    def close(self):
        self._chunk = None
