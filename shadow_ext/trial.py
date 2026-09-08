"""Evaluation timing, provenance and consolidated session summaries (no physics)."""
from __future__ import annotations

# A manual interruption stops the simulator and the webcam recorder in two
# separate terminals by hand; a few real seconds between the two is normal
# human shutdown timing, not missing evidence -- the actual task-relevant
# content (an operator watching a drop and deciding to stop) is essentially
# certain to be well within it. Applied only to the operator's END coverage
# (a few seconds of camera tail is harmless); the START side stays exact,
# since there is no equivalent reason to accept the camera starting late.
_COVERAGE_TAIL_TOLERANCE_NS = 5_000_000_000

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def provenance(root, paths):
    """Identify the working sources, including uncommitted/untracked source bytes."""
    root = Path(root)
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        'commit': git('rev-parse', 'HEAD'),
        'working_tree_status': git('status', '--porcelain', '--untracked-files=normal'),
        'source_sha256': {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
                          for p in paths},
    }


class Evaluation:
    """An explicit, one-shot interval measured on the capture's monotonic clock."""

    def __init__(self, start_mode=None, time_limit_s=None):
        if start_mode not in (None, 'manual', 'immediate'):
            raise ValueError('evaluation_start must be manual or immediate')
        if time_limit_s is not None:
            if start_mode is None or not math.isfinite(time_limit_s) or time_limit_s <= 0:
                raise ValueError('A finite positive time limit requires an evaluation start')
        self.mode = start_mode
        self.limit = time_limit_s
        self.start_ns = self.start_sim_time = None
        self.first_success_ns = self.first_success_sim_time = None

    def start(self, timestamp_ns, sim_time):
        if self.mode is None or self.start_ns is not None:
            return False
        self.start_ns, self.start_sim_time = int(timestamp_ns), float(sim_time)
        return True

    def expired(self, timestamp_ns):
        return (self.start_ns is not None and self.limit is not None and
                (timestamp_ns - self.start_ns) / 1e9 >= self.limit)

    def observe(self, success, timestamp_ns, sim_time):
        if (self.start_ns is not None and success and not self.expired(timestamp_ns)
                and self.first_success_ns is None):
            self.first_success_ns = int(timestamp_ns)
            self.first_success_sim_time = float(sim_time)

    def summary(self, end_ns, end_sim_time):
        started, succeeded = self.start_ns is not None, self.first_success_ns is not None
        return {
            'start_mode': self.mode, 'started': started, 'start_monotonic_ns': self.start_ns,
            'start_sim_time': self.start_sim_time, 'time_limit_wall_s': self.limit,
            'success_ever': succeeded, 'first_success_monotonic_ns': self.first_success_ns,
            'first_success_sim_time': self.first_success_sim_time,
            'time_to_success_wall_s': (self.first_success_ns - self.start_ns) / 1e9 if succeeded else None,
            'time_to_success_sim_s': self.first_success_sim_time - self.start_sim_time if succeeded else None,
            'elapsed_wall_s': (end_ns - self.start_ns) / 1e9 if started else None,
            'elapsed_sim_s': float(end_sim_time) - self.start_sim_time if started else None,
            'deadline_policy': 'Success observed at or after the real-time deadline is excluded',
            'task_reference': 'Task reset at evaluation start; first subsequent task update establishes references',
        }


def session_summary(session):
    """Read one session, preserving missing/incomplete evidence instead of dropping it."""
    session = Path(session).resolve()
    sim_path, operator_path = session / 'sim/metadata.json', session / 'operator/metadata.json'
    sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    operator = json.loads(operator_path.read_text()) if operator_path.exists() else {}
    evaluation = sim.get('evaluation') or {}
    problems = []
    if not sim.get('complete'):
        problems.append('simulation_incomplete_or_missing')
    if not evaluation.get('started'):
        problems.append('evaluation_not_started')
    if sim.get('trial_kind') != 'evaluation':
        problems.append('not_an_evaluation_trial')
    if sim.get('stop_reason') not in ('success', 'time_limit', 'manual_interruption'):
        problems.append('termination_' + str(sim.get('stop_reason', 'missing')))
    if not operator.get('complete'):
        problems.append('operator_incomplete_or_missing')
    if operator.get('robot') != sim.get('hand'):
        problems.append('hand_mismatch')
    if (not sim.get('boot_id') or operator.get('boot_id') != sim.get('boot_id')
            or operator.get('clock') != sim.get('clock')):
        problems.append('clock_mismatch_or_unknown')
    start = evaluation.get('start_monotonic_ns')
    # end_ns (last recorded physics state), not termination_monotonic_ns: the
    # latter includes post-interruption cleanup (flushing state chunks to
    # disk), real wall-clock time that has nothing to do with what the
    # camera needs to have covered.
    end = sim.get('end_ns', sim.get('termination_monotonic_ns'))
    if (start is None or end is None or operator.get('start_ns', math.inf) > start
            or operator.get('end_ns', -math.inf) < end - _COVERAGE_TAIL_TOLERANCE_NS):
        problems.append('operator_does_not_cover_evaluation')
    return {
        'session_id': session.name, 'session_path': str(session),
        'operator_id': sim.get('operator_id'), 'trial_kind': sim.get('trial_kind', 'legacy'),
        'model_sha256': sim.get('model_sha256'),
        'retargeter_source_sha256': (operator.get('provenance') or {}).get('source_sha256'),
        'hand': sim.get('hand'), 'task': sim.get('task'),
        'condition_id': sim.get('condition_id'), 'stop_reason': sim.get('stop_reason'),
        'evaluation': evaluation or None,
        'eligible': not problems, 'exclusion_reasons': problems,
        'simulation_complete': sim.get('complete', False),
        'operator_complete': operator.get('complete', False),
        'operator_frames': operator.get('frames'),
        'operator_dropped_frames': operator.get('dropped_frames'),
        'checkpoint_sha256': operator.get('checkpoint_sha256'),
        'conditions': {k: sim.get(k) for k in ('time_limit_wall_s', 'evaluation_start', 'integrator', 'timestep', 'gravity',
                       'manual_translation_step_m', 'manual_rotation_step_rad',
                       'slip_compensate', 'float_object', 'key_control',
                       'recv_fingers', 'recv_wrist', 'sweep_wrist', 'grasp_class',
                       'waypoint_transition_s', 'waypoint_mode', 'source_sha256')},
        'simulator_provenance': sim.get('provenance', {'commit': sim.get('simulator_commit')}),
        'retargeter_provenance': operator.get('provenance'),
        'event_counts': sim.get('event_counts', {}),
        'raw_capture_success_ever': sim.get('success_ever'),
        'final_task_metrics': sim.get('final_task_metrics'),
        'error': sim.get('error'),
    }


def consolidate(sessions):
    paths = [Path(p).resolve() for p in sessions]
    if len(paths) != len(set(paths)):
        raise ValueError('A session may appear only once')
    rows = [session_summary(p) for p in paths]
    groups = {}
    for row in rows:
        # Keep physical/control/checkpoint differences in separate result groups.
        identity = {k: row[k] for k in ('hand', 'task', 'operator_id', 'condition_id', 'checkpoint_sha256',
                                                      'model_sha256', 'retargeter_source_sha256', 'conditions')}
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(key, {'configuration': identity, 'attempts': []})
        group['attempts'].append(row)
    results = []
    for group in groups.values():
        attempts = group.pop('attempts')
        eligible = [r for r in attempts if r['eligible']]
        successes = [r for r in eligible if r['evaluation']['success_ever']]
        times = [r['evaluation']['time_to_success_wall_s'] for r in successes]
        results.append({**group, 'sessions': [r['session_id'] for r in attempts],
                        'successful_trials': len(successes), 'evaluable_trials': len(eligible),
                        'excluded_trials': len(attempts) - len(eligible),
                        'stop_reasons': dict(Counter(r['stop_reason'] for r in attempts)),
                        'success_rate': len(successes) / len(eligible) if eligible else None,
                        'median_time_to_success_wall_s': statistics.median(times) if times else None,
                        'range_time_to_success_wall_s': [min(times), max(times)] if times else None})
    return {'format_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'inclusion_rule': 'Completed evaluation with explicit start, a real attempted outcome '
                              '(success, time_limit, or manual interruption -- anything else, e.g. a '
                              'technical error, is excluded as not a valid data point), and complete '
                              'same-hand webcam covering evaluation on the same clock',
            'trials': rows, 'groups': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = consolidate(args.sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Each export gets a new name; existing results are never overwritten.
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in result['trials']:
        directory = Path(row['session_path'])
        if directory.is_dir():
            write_json(directory / 'trial_summary.json', row)
    print(f'{len(result["trials"])} trials -> {args.output}')


if __name__ == '__main__':
    main()
