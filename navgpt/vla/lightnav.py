"""HTTP bridge to an existing LightNav-0 WebSocket server; owns no model/GPU.

One bridge serves one harness episode at a time, using its own upstream connection.
Only RGB, instruction and sequence index cross into the policy process.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .context import frame_digest


def velocity_action(waypoints, stop, dt=0.1, max_linear=2.5, max_angular_dps=300.0):
    """LightNav Habitat rule: first nonzero forward/yaw row, unicycle velocity.

    The controller period belongs to execution, not to the predicted points.
    Lateral motion is ignored for the nonholonomic Habitat base.
    """
    if type(stop) is not bool:
        raise ValueError('LightNav stop must be boolean')
    if (not isinstance(waypoints, list) or len(waypoints) != 10
            or any(not isinstance(p, list) or len(p) != 3 for p in waypoints)):
        raise ValueError('LightNav must return 10 x 3 waypoints')
    if any(type(v) not in (int, float) or not math.isfinite(v) for p in waypoints for v in p):
        raise ValueError('LightNav waypoints must be finite numbers')
    row = next((p for p in waypoints if abs(p[0]) > 1e-6 or abs(p[2]) > 1e-6), waypoints[0])
    linear = min(max(row[0]/dt, 0.0), max_linear)
    angular_dps = min(max(math.degrees(row[2])/dt, -max_angular_dps), max_angular_dps)
    low_velocity = abs(linear) < 0.01 and abs(angular_dps) < 0.5
    return {'action': 'VELOCITY_CONTROL', 'policy_backend': 'lightnav',
            'stop': stop or low_velocity,
            'stop_reason': 'model_stop' if stop else 'low_velocity' if low_velocity else None,
            'action_args': {'linear_mps': linear, 'angular_rps': math.radians(angular_dps),
                            'dt_s': dt}}


class LightNavBridge:
    def __init__(self, policy_url=None, control_dt_s=0.1, max_linear_mps=2.5,
                 max_angular_dps=300.0, connect=None, inference=None):
        if inference is None and not (policy_url or '').startswith(('ws://', 'wss://')):
            raise ValueError('policy_url must use ws:// or wss://')
        for value in (control_dt_s, max_linear_mps, max_angular_dps):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('velocity configuration must be positive and finite')
        self.inference = inference
        self.policy_url = policy_url
        self.control = {'control_dt_s': control_dt_s, 'max_linear_mps': max_linear_mps,
                        'max_angular_dps': max_angular_dps}
        self.connect = connect
        self.ws = None
        self.seq = 0
        self.digest = hashlib.sha256()
        self.lock = threading.RLock()
        self.failed = False
        self.frame_ids, self.frame_digests = [], {}

    def _call(self, action, data):
        try:
            if self.inference is not None:
                return self.inference(action, data)
            if self.ws is None:
                if self.connect is None:
                    from websockets.sync.client import connect
                    self.connect = connect
                self.ws = self.connect(self.policy_url, max_size=64*1024*1024,
                                       open_timeout=30, close_timeout=10)
                self.ws.send(json.dumps({'action': 'login', 'data': {'clientId': 'navgpt-lightnav'}}))
                login = json.loads(self.ws.recv(timeout=30))['data']
                if login.get('rc') != 0:
                    raise RuntimeError('LightNav login failed')
            self.ws.send(json.dumps({'action': action, 'data': data}))
            reply = json.loads(self.ws.recv(timeout=300))['data']
            if reply.get('rc') != 0:
                raise RuntimeError('LightNav {} failed: {}'.format(action, reply.get('msg')))
            return reply
        except Exception:
            self.failed = True
            raise

    def info(self):
        with self.lock:
            if self.failed:
                raise RuntimeError('LightNav session failed; reset required')
            # A connection/login is a readiness probe, without appending an image.
            if self.ws is None and self.inference is None:
                self._call('reset', {})
            return {'model_name': 'lightnav-0', 'model_version': 'LightNav-0',
                    'backend': 'lightnav', 'policy_url': self.policy_url,
                    'execution': 'owned_vllm' if self.inference is not None else 'websocket_bridge',
                    'context_replace': False, 'history_mode': 'episode',
                    'max_context_groups': None, 'input_contract': 'lightnav_official_episode_v1',
                    'history_observe': True, 'seeded_reset': False,
                    'control': self.control, 'waypoint_count': 10,
                    'policy_camera': {'width': 480, 'height': 270, 'height_m': 0.88, 'hfov': 120.0}}

    def reset(self, data):
        if data.get('seed') is not None:
            raise ValueError('LightNav upstream does not expose seeded reset; study mode is unsupported')
        with self.lock:
            self.close()
            self.failed = False
            self._call('reset', {})
            self.seq = 0
            self.digest = hashlib.sha256()
            self.frame_ids, self.frame_digests = [], {}
            return {'status': 'reset complete', 'backend': 'lightnav'}

    def replace_context(self, data):
        raise ValueError('LightNav uses append-only episode history, not context replacement')

    def _reuse_frame(self, data):
        ident = data.get('frame_id')
        if ident is None:
            return False
        if not isinstance(ident, str) or not ident:
            raise ValueError('Frame ID must be a nonempty string')
        if ident not in self.frame_digests:
            return False
        digest = frame_digest({'views': {'policy_front': data.get('rgb_policy')}})
        if self.frame_ids[-1:] != [ident] or self.frame_digests[ident] != digest:
            raise ValueError('Cannot rewrite or reorder an observed episode frame')
        return True

    def _commit_frame(self, data, reused):
        ident = data.get('frame_id')
        if ident is not None and not reused:
            self.frame_ids.append(ident)
            self.frame_digests[ident] = frame_digest({'views': {'policy_front': data['rgb_policy']}})

    def _next(self, image, instruction, reused=False):
        if self.failed:
            raise RuntimeError('LightNav session failed; reset required')
        if not isinstance(image, str) or not image:
            raise ValueError('rgb_policy image is required')
        if reused and self.inference is None:
            raise ValueError('Reusing an endpoint requires the owned LightNav inference service')
        data = {'seq': self.seq, 'image': image, 'instruction': instruction}
        reply = self._call('predict' if reused else 'next', data)
        if not reused:
            self.digest.update(image.encode())
            self.seq += 1
        return reply

    def observe(self, data):
        with self.lock:
            reused = self._reuse_frame(data)
            if not reused:
                self._next(data.get('rgb_policy'), '')
                self._commit_frame(data, reused)
            return {'status': 'observed', **self.state()}

    def act(self, data):
        text = data.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('nonempty text instruction is required')
        with self.lock:
            reused = self._reuse_frame(data)
            reply = self._next(data.get('rgb_policy'), text, reused)
            try:
                waypoints = reply['actions']['actions']
                action = velocity_action(waypoints, reply['stop'], self.control['control_dt_s'],
                                         self.control['max_linear_mps'], self.control['max_angular_dps'])
            except Exception:
                self.failed = True
                raise
            action['raw_text'] = reply.get('raw_text')
            if reply.get('decode_error'):
                action['decode_error'] = reply['decode_error']
                action['stop_reason'] = 'decode_error'
            action['latency_ms'] = reply.get('latency_ms')
            if reply.get('timings_ms'):
                action['inference_metrics'] = reply['timings_ms']
            self._commit_frame(data, reused)
            return [action, waypoints]

    def state(self):
        with self.lock:
            return {'history_frames': self.seq, 'history_sha256': self.digest.hexdigest(),
                    'history_mode': 'episode', 'frame_ids': list(self.frame_ids),
                    'failed': self.failed, 'backend': 'lightnav', 'seeded_reset': False}

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None


def main(options):
    options = dict(options)
    host, port = options.pop('host'), options.pop('port')
    options.pop('result_path', None)
    service = LightNavBridge(**options)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == '/info': self._send(service.info())
                elif self.path == '/state': self._send(service.state())
                else: self._send({'error': 'not found'}, 404)
            except Exception as exc:
                self._send({'error': str(exc)}, 502)

        def do_POST(self):
            methods = {'/reset': service.reset, '/observe': service.observe, '/act': service.act,
                       '/context': service.replace_context}
            if self.path not in methods:
                return self._send({'error': 'not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 64*1024*1024:
                    raise ValueError('invalid request size')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict): raise ValueError('request must be an object')
                self._send(methods[self.path](data))
            except (ValueError, TypeError) as exc:
                self._send({'error': str(exc)}, 400)
            except Exception as exc:
                self._send({'error': str(exc)}, 502)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        service.close()
        server.server_close()
