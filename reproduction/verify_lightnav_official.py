"""Explicit GPU parity probe; never starts a benchmark or changes a simulator.

Run in the dedicated LightNav environment with one visible GPU. Replays recorded
policy-camera images through both the official Habitat policy and our owned,
batched adapter, sharing one model. Requires a continuous frame-000000... source.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time


def fingerprint(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    return value


async def verify(args):
    import numpy as np
    from PIL import Image
    from lightnav.habitat.policy import TrajVocabVLNCEPolicy, select_action_waypoint
    from lightnav.inference.samples import build_vln_traj_sample
    from lightnav.velocity import first_waypoint_to_velocity_cmd
    from navgpt.vla.shared import BackendFactory
    from navgpt.vla.lightnav import velocity_action
    from navgpt.planner.tools.vla_client import NavGPTVLAClient

    factory = BackendFactory({'backend': 'lightnav', 'model_path': args.model,
                              'max_batch_size': 2, 'max_wait_ms': 2,
                              'model_options': {'gpu_memory_utilization': .3, 'max_num_seqs': 2}})
    report = {'started': time.time(), 'checks': [], 'controller_cases': 0}
    await factory.start()
    try:
        owned = factory.make()
        engine = factory.service.engine
        reference = TrajVocabVLNCEPolicy(
            engine, centroids=factory.service.centroids,
            rvq_bundle=factory.service.rvq_bundle, dt=.1,
            lin_vel_range=(0., 2.5), ang_vel_range=(-300., 300.),
            num_history_frames=factory.service.num_history_frames,
        )
        instruction = Path(args.instruction).read_text().strip()
        reference.reset({'instruction': instruction})

        def transport(method, path, data=None):
            if method == 'GET':
                return {'/info': owned.info, '/state': owned.state}[path]()
            return {'/observe': owned.observe, '/act': owned.act,
                    '/reset': owned.reset, '/context': owned.replace_context}[path](data)

        client = NavGPTVLAClient('', backend='lightnav', transport=transport)
        await factory.run(client.reset)
        frames = []
        picks = {0, 1, 15, 16, 33, 89, 128, args.frames - 1}
        for i in range(args.frames):
            path = Path(args.observations) / f'frame-{i:06d}' / 'policy_front.png'
            image = np.array(Image.open(path).convert('RGB'))
            frames.append({'id': f'frame-{i:06d}',
                           'views': {'policy_front': base64.b64encode(path.read_bytes()).decode()}})
            if i not in picks:
                reference.observe({'rgb': image})
                continue
            official_action = await asyncio.to_thread(reference.act, {'rgb': image})

            def step():
                client.prepare_context('task-1', frames)
                return client.act(frames[-1]['views'], instruction, 'parity', None, None,
                                  frame_id=frames[-1]['id'])

            actual = await factory.run(step)
            agent = owned.inference.agent
            expected_sample = fingerprint(build_vln_traj_sample(
                reference.agent._get_video_tensor(), instruction,
                reference.agent._history_frame_ids, engine.bundle))
            actual_sample = fingerprint(factory.service._build_sample_for_session(agent))
            np.testing.assert_allclose(actual['navigation'], reference._last_waypoints, atol=1e-6)
            assert actual['raw_action']['raw_text'] == reference._last_raw_text
            assert actual_sample == expected_sample
            assert agent._history_frame_ids == list(range(i + 1))
            raw = official_action['action_args']
            expected_v = (raw['linear_velocity'] + 1) * 1.25
            expected_w = np.deg2rad(-300 + (raw['angular_velocity'] + 1) * 300)
            np.testing.assert_allclose([actual['velocity']['linear_mps'], actual['velocity']['angular_rps']],
                                       [expected_v, expected_w], atol=1e-6)
            report['checks'].append({'frame': i, 'buffer_frames': len(agent._history_frame_ids),
                                     'raw_text': reference._last_raw_text,
                                     'sample': actual_sample, 'matched': True})
            print(f'parity frame={i} history={len(agent._history_frame_ids)} matched', flush=True)

        # A new instruction at the same endpoint must not fabricate a timestep.
        revised = 'Turn left toward the doorway'
        expected, raw, _ = await asyncio.to_thread(reference.agent.predict_waypoints, revised, 'vlnce_traj')
        def resume():
            client.prepare_context('task-2', frames)
            return client.act(frames[-1]['views'], revised, 'parity', None, None, frame_id=frames[-1]['id'])
        actual = await factory.run(resume)
        np.testing.assert_allclose(actual['navigation'], expected, atol=1e-6)
        assert actual['raw_action']['raw_text'] == raw
        assert owned.state()['history_frames'] == args.frames
        report['same_endpoint_reinstruction'] = True

        # Two simultaneous sessions must retain private histories and match the
        # serial official reference. Synchronize dispatch to exercise batching.
        import threading
        barrier = threading.Barrier(2)
        paired = []
        for offset, count, text in [(0, 33, instruction), (100, 90, revised)]:
            other = factory.make()
            other.reset({})
            ref = TrajVocabVLNCEPolicy(
                engine, centroids=factory.service.centroids,
                rvq_bundle=factory.service.rvq_bundle, dt=.1,
                lin_vel_range=(0., 2.5), ang_vel_range=(-300., 300.),
                num_history_frames=factory.service.num_history_frames,
            )
            ref.reset({'instruction': text})
            for j in range(offset, offset + count):
                path = Path(args.observations) / f'frame-{j:06d}' / 'policy_front.png'
                rgb = np.array(Image.open(path).convert('RGB'))
                encoded = base64.b64encode(path.read_bytes()).decode()
                if j < offset + count - 1:
                    ref.observe({'rgb': rgb})
                    other.observe({'rgb_policy': encoded, 'frame_id': str(j)})
            await asyncio.to_thread(ref.act, {'rgb': rgb})
            paired.append((other, ref, {'rgb_policy': encoded, 'frame_id': str(j), 'text': text}))
        def simultaneous(item):
            actor, _, payload = item
            barrier.wait(timeout=30)
            return actor.act(payload)
        outputs = await asyncio.gather(*(factory.run(lambda item=item: simultaneous(item)) for item in paired))
        for (actor, ref, _), (action, waypoints) in zip(paired, outputs):
            np.testing.assert_allclose(waypoints, ref._last_waypoints, atol=1e-6)
            assert action['raw_text'] == ref._last_raw_text
            assert actor.inference.agent._history_frame_ids == ref.agent._history_frame_ids
        report['concurrent_sessions'] = [
            {'history_frames': actor.state()['history_frames'],
             'batch_size': output[0]['inference_metrics']['batch_size'], 'matched': True}
            for (actor, _, _), output in zip(paired, outputs)
        ]

        # Numeric control conversion, including STOP, rotation-only and clipping.
        rng = np.random.default_rng(42)
        cases = [np.zeros((10, 3), dtype=np.float32)]
        cases += [rng.normal(0, 1, (10, 3)).astype(np.float32) for _ in range(100)]
        for wp in cases:
            _, row = select_action_waypoint(wp)
            raw = first_waypoint_to_velocity_cmd(row, .1, (0., 2.5), (-300., 300.))['action_args']
            ours = velocity_action(wp.tolist(), bool(np.allclose(wp, 0)))
            np.testing.assert_allclose(
                [ours['action_args']['linear_mps'], ours['action_args']['angular_rps']],
                [(raw['linear_velocity'] + 1) * 1.25,
                 np.deg2rad(-300 + (raw['angular_velocity'] + 1) * 300)], atol=1e-6)
            report['controller_cases'] += 1
        from unittest.mock import patch
        with patch.object(reference.engine, 'generate_from_frames', return_value=('invalid', 0)):
            fallback = reference.act({'rgb': image})
        zero, text = agent.decode_waypoints('invalid')
        assert not zero.any() and text == 'invalid' and agent.decode_error
        assert fallback == reference.stop_action()
        report['decode_failure_zero_stop'] = True
        report['native_input'] = factory.metadata['native_input']
        report['passed'] = True
    finally:
        report['finished'] = time.time()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        await factory.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--output', required=True)
    asyncio.run(verify(parser.parse_args()))
