"""Validation shared by the two backend context replacement interfaces."""
import hashlib
import json


def context_frames(data, cameras, cached=None):
    if not isinstance(data.get('context_key'), str) or not data['context_key']:
        raise ValueError('context_key is required')
    frames = data.get('frames')
    # /act appends the current frame; replacement contains only its past.
    if not isinstance(frames, list) or len(frames) > 15:
        raise ValueError('Context replacement allows at most 15 past frame groups')
    ids = []
    resolved = []
    for frame in frames:
        ident, images = frame.get('id'), frame.get('views')
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('Frame IDs must be nonempty and unique')
        if images is None:
            images = (cached or {}).get(ident)
        if not isinstance(images, dict) or set(images) != set(cameras):
            raise ValueError('Context contains incompatible cameras')
        if not all(isinstance(value, str) and value for value in images.values()):
            raise ValueError('Images must be nonempty base64 strings')
        ids.append(ident)
        resolved.append({'id': ident, 'views': images})
    return resolved


def frame_digest(frame):
    return hashlib.sha256(json.dumps(frame['views'], sort_keys=True).encode()).hexdigest()


class AgentContext:
    """Maintain ID-to-image alignment without rerunning cached vision preprocessing."""
    cameras = ('front', 'right', 'back', 'left')

    def __init__(self, agent, decode):
        self.agent, self.decode = agent, decode
        self.reset()

    def reset(self):
        self.key, self.ids, self.digests, self.images = None, [], {}, {}

    def replace(self, data):
        frames = context_frames(data, self.cameras, self.images)
        old = {ident: i for i, ident in enumerate(self.ids)}
        if len(self.agent.rgb_lists[0]) != len(self.ids):
            old = {}
        groups, reuse = [], []
        for frame in frames:
            index = old.get(frame['id'])
            if index is not None and self.digests.get(frame['id']) == frame_digest(frame):
                groups.append([history[index] for history in self.agent.rgb_lists])
                reuse.append(index)
            else:
                groups.append([self.decode(frame['views'][cam]) for cam in self.cameras])
                reuse.append(None)
        self.agent.replace_history(groups, reuse)
        self.key = data['context_key']
        self.ids = [frame['id'] for frame in frames]
        self.digests = {frame['id']: frame_digest(frame) for frame in frames}
        self.images = {frame['id']: frame['views'] for frame in frames}
        return self.state()

    def validate_act(self, data):
        if data.get('context_key') != self.key or self.key is None:
            raise ValueError('Replace context before acting on this task')
        ident = data.get('frame_id')
        if not isinstance(ident, str) or not ident or ident in self.ids or len(self.ids) >= 16:
            raise ValueError('Current frame must be new and context must fit 16 groups')

    def commit_act(self, data):
        frame = {'views': {cam: data['rgb_' + cam] for cam in self.cameras}}
        self.ids.append(data['frame_id'])
        self.digests[data['frame_id']] = frame_digest(frame)
        self.images[data['frame_id']] = frame['views']

    def state(self):
        return {'context_key': self.key, 'frame_ids': list(self.ids), 'history_frames': len(self.ids)}
