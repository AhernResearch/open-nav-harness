import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from navgpt.settings import load
from navgpt.vla.lightnav import LightNavBridge, velocity_action
from navgpt.planner.tools.vla_client import NavGPTVLAClient

ROOT = Path(__file__).resolve().parents[1]


class FakeWS:
    def __init__(self):
        self.sent, self.replies = [], []
        self.closed = False
        self.prediction = {'rc': 0, 'stop': False,
                           'actions': {'actions': [[0., 0., 0.]] + [[.2, .03, .1]]*9}}

    def send(self, value):
        message = json.loads(value)
        self.sent.append(message)
        self.replies.append(self.prediction if message['action'] == 'next'
                            and message['data'].get('instruction') else {'rc': 0})

    def recv(self, timeout=None):
        return json.dumps({'data': self.replies.pop(0)})

    def close(self):
        self.closed = True


class LightNavTests(unittest.TestCase):
    def test_native_velocity_rule_and_clipping(self):
        value = velocity_action([[0, 0, 0]]+[[.4, 99, math.pi]]*9, False)
        self.assertFalse(value['stop'])
        self.assertEqual(value['action_args']['linear_mps'], 2.5)
        self.assertAlmostEqual(value['action_args']['angular_rps'], math.radians(300))
        self.assertEqual(value['action_args']['dt_s'], .1)

    def test_model_and_zero_velocity_stop_are_explicit(self):
        self.assertEqual(velocity_action([[0, 0, 0]]*10, False)['stop_reason'], 'low_velocity')
        self.assertEqual(velocity_action([[.2, 0, 0]]*10, True)['stop_reason'], 'model_stop')

    def test_invalid_waypoints_never_become_stop(self):
        for points, stop in [([], False), ([[0, 0, 0]]*8, False),
                             ([[float('nan'), 0, 0]]*10, False), ([[0, 0, 0]]*10, 1)]:
            with self.subTest(points=points):
                with self.assertRaises(ValueError): velocity_action(points, stop)

    def test_session_history_instruction_change_and_reset(self):
        connections = []
        def connect(*args, **kwargs):
            ws = FakeWS(); connections.append(ws); return ws
        service = LightNavBridge('ws://local:8050', connect=connect)
        service.reset({})
        service.observe({'rgb_policy': 'repair'})
        service.act({'rgb_policy': 'current', 'text': 'first task', 'global_gps': [999]*3})
        service.act({'rgb_policy': 'next', 'text': 'revised task'})
        messages = [m['data'] for m in connections[0].sent if m['action'] == 'next']
        self.assertEqual([m['seq'] for m in messages], [0, 1, 2])
        self.assertEqual([m['instruction'] for m in messages], ['', 'first task', 'revised task'])
        self.assertTrue(all(set(m) == {'seq', 'image', 'instruction'} for m in messages))
        service.reset({})
        self.assertTrue(connections[0].closed)
        self.assertEqual(service.state()['history_frames'], 0)
        self.assertEqual(len(connections), 2)
        with self.assertRaises(ValueError): service.reset({'seed': 7})
        service.close()

    def test_corrupt_reply_requires_reset(self):
        ws = FakeWS()
        ws.prediction['actions']['actions'] = []
        service = LightNavBridge('ws://local:8050', connect=lambda *a, **kw: ws)
        service.reset({})
        with self.assertRaises(ValueError): service.act({'rgb_policy': 'frame', 'text': 'go'})
        with self.assertRaises(RuntimeError): service.act({'rgb_policy': 'frame', 'text': 'go'})
        self.assertTrue(service.state()['failed'])

    def test_lightnav_http_payload_contains_no_pose_or_panorama(self):
        action = velocity_action([[.2, 0, 0]]*10, False)
        class Response:
            def raise_for_status(self): pass
            def json(self): return [action, [[.2, 0, 0]]*10]
        client = NavGPTVLAClient('http://local', backend='lightnav')
        with patch('requests.post', return_value=Response()) as post:
            advice = client.act({'policy_front': 'RGB'}, 'go', 'episode', None, None)
        self.assertEqual(set(post.call_args.kwargs['json']), {'rgb_policy', 'text', 'episode_id'})
        self.assertIn('velocity', advice)
        self.assertNotIn('position', advice)
        self.assertFalse(advice['should_stop'])

    def test_original_navgpt_stop_still_requires_two_hits(self):
        class ArrayLike:
            def __iter__(self): return iter([0, 0, 0])
            def __getitem__(self, index): return [0, 0, 0][index]
            def __bool__(self): raise ValueError('array truth is ambiguous')
        class Response:
            def raise_for_status(self): pass
            def json(self): return [{}, [[0, 0, 0]]*8]
        client = NavGPTVLAClient('http://local')
        with patch('requests.post', return_value=Response()):
            first = client.act(dict.fromkeys(('front','right','back','left'), 'RGB'), 'go', '1', ArrayLike(), [1,0,0,0])
            second = client.act(dict.fromkeys(('front','right','back','left'), 'RGB'), 'go', '1', [0]*3, [1,0,0,0])
        self.assertFalse(first['should_stop'])
        self.assertTrue(second['should_stop'])

    def test_settings_resolve_external_policy_without_checkpoint(self):
        settings = load(ROOT/'configs/experiments/lightnav/planner_r2r.json',
                        ROOT/'configs/paths.lightnav.example.json')
        cfg = settings.run_config()
        options = settings.vla_options()
        self.assertEqual(cfg.vla_backend, 'lightnav')
        self.assertNotIn('model_path', options)
        env = settings.environment_options()['config']
        self.assertEqual(env['policy_camera']['height_m'], .88)
        self.assertEqual(env['velocity_control']['control_dt_s'], options['control_dt_s'])

    def test_repair_history_is_causal_and_endpoint_not_duplicated(self):
        from navgpt.planner.tools.session import NavigationSession
        tools = NavigationSession()
        class Policy:
            def __init__(self): self.frames = []
            def observe_history(self, views): self.frames.append(views['policy_front'])
        policy = Policy()
        def frame(i): return {'policy_frame_id': i, 'views': {'policy_front': str(i)}}
        with patch.object(tools, 'VLA_BACKEND', 'lightnav'), patch.object(tools, '_vla', policy), patch.object(tools, '_history_pending', None):
            tools._feed_passive_history([frame(1), frame(2)])
            tools._feed_passive_history([frame(3), frame(4)])
            self.assertEqual(policy.frames, ['1', '2', '3'])
            tools._prepare_policy_observation({'policy_frame_id': 4})
            self.assertEqual(policy.frames, ['1', '2', '3'])
            self.assertIsNone(tools._history_pending)

    def test_capped_endpoint_is_retained_but_inferred_stop_is_not_duplicated(self):
        from navgpt.planner.tools.session import NavigationSession
        tools = NavigationSession()
        class Policy:
            def __init__(self): self.frames = []
            def observe_history(self, views): self.frames.append(views['policy_front'])
        policy = Policy()
        def pano(i): return {'policy_frame_id': i, 'raw_views': {'policy_front': str(i)}}
        def frame(i): return {'policy_frame_id': i, 'views': {'policy_front': str(i)}}
        with patch.object(tools, 'VLA_BACKEND', 'lightnav'), patch.object(tools, '_vla', policy), patch.object(tools, '_history_pending', None), patch.object(tools, '_last_policy_inference_id', 0):
            tools._stage_policy_observation(pano(1))
            tools._feed_passive_history([frame(2), frame(3)])
            self.assertEqual(policy.frames, ['1', '2'])
            tools._prepare_policy_observation(pano(3))
            tools._last_policy_inference_id = 3
            tools._stage_policy_observation(pano(3))
            self.assertIsNone(tools._history_pending)
            self.assertEqual(policy.frames, ['1', '2'])


if __name__ == '__main__': unittest.main()
