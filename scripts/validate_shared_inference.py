"""Replay fixed development observations at concurrency 1/2/4; never move a robot.

Manifest: [{name, instruction, frames: [{policy_front, front, right, back, left}]}].
All frame values are local image paths. At least three frames per case are required.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid
import httpx


async def validate(url, backend, manifest):
    cases = json.loads(Path(manifest).read_text())
    cameras = ('policy_front',) if backend == 'lightnav' else ('front', 'right', 'back', 'left')
    frames = [[{'id': str(i), 'views': {
        cam: base64.b64encode(Path(frame[cam]).read_bytes()).decode() for cam in cameras
    }} for i, frame in enumerate(case['frames'])] for case in cases]
    hashes = [[{cam: hashlib.sha256(base64.b64decode(frame['views'][cam])).hexdigest()
                for cam in cameras} for frame in group] for group in frames]
    report = {'backend': backend, 'cases': [c['name'] for c in cases], 'frame_hashes': hashes, 'runs': []}
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=300) as http:
        report['service'] = (await http.get('/info')).json()
        async def post(path, data):
            response = await http.post(path, json=data)
            response.raise_for_status()
            return response.json()
        async def operation(ident, name, data):
            return (await post(f'/sessions/{ident}/call', {
                'request_id': uuid.uuid4().hex, 'operation': name, 'data': data,
            }))['result']
        async def sequence(ident, index):
            replies = []
            for step in range(1, 3):
                await operation(ident, 'replace_context', {'context_key': 'task', 'frames': frames[index][:step]})
                current = frames[index][step]
                data = {'context_key': 'task', 'frame_id': current['id'], 'text': cases[index]['instruction'],
                        'global_gps': [0, 0, 0], 'global_compass': [1, 0, 0, 0], 'is_stuck': False}
                if backend == 'lightnav':
                    data['rgb_policy'] = current['views']['policy_front']
                else:
                    data.update({'rgb_' + cam: current['views'][cam] for cam in cameras})
                started = time.monotonic()
                result = await operation(ident, 'act', data)
                replies.append({'latency_s': time.monotonic() - started, 'result': result})
            state = await operation(ident, 'state', {})
            return {'replies': replies, 'state': state}
        # Warmup, excluded from measurements.
        ident = uuid.uuid4().hex
        await post('/sessions', {'session_id': ident, 'backend': backend})
        await operation(ident, 'reset', {'seed': 7} if backend == 'navgpt' else {})
        await sequence(ident, 0)
        await post(f'/sessions/{ident}/close', {})
        for concurrency in (1, 2, 4):
            ids = [uuid.uuid4().hex for _ in cases]
            for ident in ids:
                await post('/sessions', {'session_id': ident, 'backend': backend})
                await operation(ident, 'reset', {'seed': 7} if backend == 'navgpt' else {})
            sem = asyncio.Semaphore(concurrency)
            async def run(i):
                async with sem:
                    return await sequence(ids[i], i)
            started = time.monotonic()
            results = await asyncio.gather(*(run(i) for i in range(len(cases))))
            wall_s = time.monotonic() - started
            for ident in ids:
                await post(f'/sessions/{ident}/close', {})
            times = sorted(r['latency_s'] for case in results for r in case['replies'])
            row = {'concurrency': concurrency, 'wall_s': wall_s, 'requests_per_s': len(times)/wall_s,
                   'p50_s': statistics.median(times), 'p95_s': times[max(0, int(.95*len(times)+.999)-1)],
                   'results': results}
            report['runs'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'results'}), flush=True)
    baseline = report['runs'][0]['results']
    for run in report['runs']:
        errors = []
        max_error = 0.0
        for i, case in enumerate(run['results']):
            if case['state'] != baseline[i]['state']:
                errors.append(f'state differs for case {i}')
            for j, reply in enumerate(case['replies']):
                a, b = reply['result'], baseline[i]['replies'][j]['result']
                delta = max(abs(x-y) for p,q in zip(a[1],b[1]) for x,y in zip(p,q))
                max_error = max(max_error, delta)
                if len(a[1]) != len(b[1]) or delta > (0 if backend == 'lightnav' else 1e-4):
                    errors.append(f'waypoints differ for case {i}, step {j}')
                if backend == 'lightnav' and (a[0].get('raw_text') != b[0].get('raw_text') or a[0].get('action') != b[0].get('action')):
                    errors.append(f'tokens/stop differ for case {i}, step {j}')
        run['parity_errors'], run['max_waypoint_error'] = errors, max_error
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--backend', required=True, choices=['lightnav','navgpt'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(args.url,args.backend,args.manifest))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if any(r['parity_errors'] for r in result['runs']):
        raise SystemExit('Parity failed; inspect report before deployment')


if __name__=='__main__':
    main()
