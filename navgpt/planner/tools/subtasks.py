"""Episode-local task lifecycle and causal VLA context (no simulator imports)."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path


class SubtaskManager:
    def __init__(self, instruction, directory=None, max_frames=16, boundary_frames=2):
        self.instruction = instruction
        self.directory = Path(directory) if directory else None
        self.max_frames, self.boundary_frames = max_frames, boundary_frames
        self.tasks, self.revision, self.summary = [], 0, ''
        self.frames, self.motion_version = [], 0
        self.map_ref, self.map_version = None, -1
        self.needs_review, self.needs_replan = False, False
        self.context_key, self.anchor, self.anchor_task_id = None, None, None
        self.last_review, self.aborted = None, False
        self._last_fingerprint = None
        self._event_number = 0

    def _event(self, kind, **fields):
        self._event_number += 1
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / 'events.jsonl').open('a') as out:
                out.write(json.dumps({'event': self._event_number, 'kind': kind, **fields},
                                     ensure_ascii=False) + '\n')
            state = self.directory / 'state.json'
            temporary = state.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + '\n')
            temporary.replace(state)

    def snapshot(self):
        return {'revision': self.revision, 'tasks': copy.deepcopy(self.tasks),
                'summary': self.summary, 'needs_review': self.needs_review,
                'needs_replan': self.needs_replan, 'context_key': self.context_key,
                'current_frame_id': self.frames[-1]['id'] if self.frames else None,
                'current_map_id': self.map_ref if self.map_version == self.motion_version else None,
                'last_review': self.last_review, 'aborted': self.aborted}

    def observe(self, views, pose, steps):
        if not views:
            raise ValueError('Subtask mode requires raw policy observations')
        fingerprint = hashlib.sha256(json.dumps(
            [views, self.motion_version], sort_keys=True).encode()).hexdigest()
        if fingerprint == self._last_fingerprint:
            if pose:
                self.frames[-1]['pose'] = pose
            self.frames[-1]['steps'] = steps
            return self.frames[-1]['id']
        frame_id = 'frame-{:06d}'.format(len(self.frames))
        frame = {'id': frame_id, 'views': dict(views), 'pose': pose,
                 'steps': steps, 'motion_version': self.motion_version}
        self.frames.append(frame)
        self._last_fingerprint = fingerprint
        refs = {}
        if self.directory:
            folder = self.directory / 'observations' / frame_id
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in views.items():
                if name not in ('front', 'right', 'back', 'left', 'policy_front'):
                    raise ValueError('Unknown camera')
                path = folder / (name + '.png')
                raw = base64.b64decode(value, validate=True)
                path.write_bytes(raw)
                refs[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        self._event('observation', frame_id=frame_id, images=refs,
                    pose=pose, steps=steps, context_key=self.context_key)
        return frame_id

    def mark_motion(self):
        self.motion_version += 1
        self.needs_review = True

    def mark_map(self):
        if not self.frames or self.frames[-1]['motion_version'] != self.motion_version:
            raise ValueError('Refresh camera evidence before recording a map')
        self.map_ref = 'map-' + self.frames[-1]['id'].split('-', 1)[1]
        self.map_version = self.motion_version
        self._event('map', map_id=self.map_ref, frame_id=self.frames[-1]['id'])

    def update_plan(self, subtasks, reason, reopen_ids=None):
        if not self.frames:
            raise ValueError('Observe the first frame before planning')
        if self.needs_review:
            raise ValueError('Review the current execution before changing the plan')
        if not isinstance(subtasks, list) or not 1 <= len(subtasks) <= 8:
            raise ValueError('Plan needs 1-8 sequential subtasks')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A plan update needs a reason')
        old = {t['id']: t for t in self.tasks}
        reopen = set(reopen_ids or [])
        if not reopen <= set(old):
            raise ValueError('Cannot reopen an unknown subtask')
        new, ids = [], set()
        for source in subtasks:
            task = {k: source.get(k) for k in ('id', 'source_quote', 'instruction', 'completion_criterion')}
            ident = task['id']
            if type(ident) is not int or ident < 1 or ident in ids:
                raise ValueError('Subtask IDs must be distinct positive integers')
            ids.add(ident)
            for key in ('source_quote', 'instruction', 'completion_criterion'):
                if not isinstance(task[key], str) or not task[key].strip():
                    raise ValueError('Missing subtask ' + key)
            if task['source_quote'] not in self.instruction:
                raise ValueError('source_quote must be an exact substring of the original instruction')
            previous = old.get(ident)
            if previous and previous['status'] == 'accepted' and ident not in reopen:
                if any(task[k] != previous[k] for k in task):
                    raise ValueError('Explicitly reopen an accepted subtask before editing it')
                task['status'] = 'accepted'
            else:
                task['status'] = 'pending'
            new.append(task)
        if any(t['status'] == 'accepted' and t['id'] not in ids for t in self.tasks):
            raise ValueError('Accepted subtasks cannot disappear from plan history')
        accepted_seen_pending = False
        for task in new:
            if task['status'] != 'accepted':
                accepted_seen_pending = True
            elif accepted_seen_pending:
                raise ValueError('Reopen dependent later subtasks when reopening an earlier one')
        # Refining the remaining command does not turn this same subtask's past
        # into previous-task history. Keep its anchor unless its identity changes
        # or the caller explicitly requests a restart.
        continuing = next((t for t in new if t['status'] != 'accepted'), None)
        previous = old.get(self.anchor_task_id)
        if (continuing is None or previous is None
                or continuing['id'] != self.anchor_task_id
                or continuing['id'] in reopen
                or continuing['source_quote'] != previous['source_quote']):
            self.anchor, self.anchor_task_id = None, None
        self.tasks = new
        self.revision += 1
        self.needs_replan = False
        self.context_key = None
        self._event('plan', reason=reason, reopen_ids=sorted(reopen), plan=self.snapshot())
        return self.snapshot()

    def current(self):
        if not self.tasks:
            raise ValueError('Submit a plan before moving')
        if self.aborted:
            raise ValueError('Task was aborted')
        task = next((t for t in self.tasks if t['status'] != 'accepted'), None)
        if task is None:
            raise ValueError('All subtasks accepted; stop the episode')
        return task

    def begin(self):
        if self.needs_review:
            raise ValueError('Review returned execution before moving again')
        if self.needs_replan:
            raise ValueError('Submit the revised plan before moving')
        task = self.current()
        key = '{}:{}'.format(self.revision, task['id'])
        if key != self.context_key:
            self.context_key = key
            if self.anchor is None or self.anchor_task_id != task['id']:
                self.anchor = len(self.frames) - 1
                self.anchor_task_id = task['id']
        task['status'] = 'active'
        self._event('begin', task_id=task['id'], context_key=key,
                    anchor_frame_id=self.frames[self.anchor]['id'])
        return task

    def selected_frames(self):
        if self.anchor is None:
            raise ValueError('No active subtask context')
        if self.max_frames == 1:
            return self.frames[-1:]
        essential = 1 if self.anchor == len(self.frames) - 1 else 2
        boundary_count = min(self.boundary_frames, self.max_frames - essential)
        boundary = list(range(max(0, self.anchor - boundary_count), self.anchor))
        capacity = self.max_frames - len(boundary)
        active = [self.anchor]
        active += list(range(max(self.anchor + 1, len(self.frames) - capacity + 1), len(self.frames)))
        indices = boundary + active
        return [self.frames[i] for i in indices]

    def policy_instruction(self):
        task = self.current()
        background = ('\nCompleted navigation history (background only; do not repeat): '
                      + self.summary) if self.summary else ''
        return (task['instruction']
                + '\nStop this local operation when: ' + task['completion_criterion']
                + background)

    def returned(self, reason):
        self.needs_review = True
        self._event('return', reason=reason)

    def review(self, outcome, evidence_refs, summary, reason):
        task = self.current()
        if outcome not in ('accepted', 'continue', 'replan'):
            raise ValueError('outcome must be accepted, continue, or replan')
        if not isinstance(summary, str) or len(summary) > 800:
            raise ValueError('Summary must contain at most 800 characters')
        if not summary.strip() or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Review needs a nonempty evidence-based summary and reason')
        if not isinstance(evidence_refs, list) or not all(isinstance(x, str) for x in evidence_refs):
            raise ValueError('evidence_refs must list observed frame and map IDs')
        current = self.frames[-1]
        if current['motion_version'] != self.motion_version or self.map_version != self.motion_version:
            raise ValueError('Fresh camera and map evidence are required after motion')
        if not {current['id'], self.map_ref} <= set(evidence_refs):
            raise ValueError('Cite current_frame_id and current_map_id from the latest tool state')
        known = {f['id'] for f in self.frames} | {'map-' + f['id'].split('-', 1)[1] for f in self.frames}
        if not set(evidence_refs) <= known:
            raise ValueError('Unknown evidence reference')
        task['status'] = 'accepted' if outcome == 'accepted' else 'active'
        self.needs_review = False
        self.needs_replan = outcome == 'replan'
        self.summary = summary
        self.last_review = {'task_id': task['id'], 'outcome': outcome,
                            'evidence_refs': evidence_refs, 'reason': reason}
        self._event('review', **self.last_review, summary=summary)
        return self.snapshot()

    def finish(self, status):
        if status not in ('complete', 'abort'):
            raise ValueError('status must be complete or abort')
        if status == 'complete':
            if not self.tasks or any(t['status'] != 'accepted' for t in self.tasks):
                raise ValueError('Accept every required subtask before declaring completion')
            if self.needs_review or self.map_version != self.motion_version:
                raise ValueError('Final position must have fresh reviewed evidence')
        self.aborted = status == 'abort'
        self._event('finish', status=status)

    def record_context(self, frames, instruction):
        self._event('policy_context', context_key=self.context_key,
                    frame_ids=[f['id'] for f in frames], instruction=instruction,
                    summary=self.summary)
