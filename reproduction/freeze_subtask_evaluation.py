"""Freeze scene-disjoint paired evaluation before subtask harness tuning."""
import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--previous-summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise SystemExit('Frozen manifest exists; refusing to replace it')
    source = Path(args.dataset)
    raw = source.read_bytes()
    episodes = json.loads(gzip.decompress(raw))['episodes']
    previous = json.loads(Path(args.previous_summary).read_text())
    dev_scenes = {e['scene_id'] for e in previous['episodes']}
    groups = collections.defaultdict(list)
    for index, ep in enumerate(episodes):
        if ep['scene_id'] in dev_scenes:
            continue
        route = (ep['scene_id'], tuple(ep['start_position']),
                 tuple(tuple(g['position']) for g in ep['goals']),
                 json.dumps(ep.get('reference_path', []), separators=(',', ':')))
        groups[route].append((index, ep))
    scenes = sorted({r[0] for r in groups})
    rng = random.Random(20261107)
    pools = {}
    for scene in scenes:
        routes = [g for r, g in groups.items() if r[0] == scene]
        routes.sort(key=lambda g: float(g[0][1].get('info', {}).get('geodesic_distance', 0)))
        # Round-robin within length thirds keeps short, medium and long routes.
        thirds = [routes[i * len(routes) // 3:(i + 1) * len(routes) // 3]
                  for i in range(3)]
        for bucket in thirds:
            rng.shuffle(bucket)
        order = []
        while any(thirds):
            for bucket in thirds:
                if bucket:
                    order.append(bucket.pop())
        pools[scene] = order
    chosen = []
    while len(chosen) < 50:
        for scene in scenes:
            if pools[scene] and len(chosen) < 50:
                candidates = pools[scene].pop(0)
                index, ep = rng.choice(candidates)
                chosen.append({'index': index, 'episode_id': str(ep['episode_id']),
                               'scene_id': scene,
                               'instruction': ep['instruction']['instruction_text'],
                               'route_group_indices': [i for i, _ in candidates]})
        if not any(pools.values()) and len(chosen) < 50:
            raise ValueError('Fewer than 50 eligible route groups')
    manifest = {'schema': 'subtask_paired_eval_v1', 'seed': 20261107,
                'dataset_sha256': hashlib.sha256(raw).hexdigest(),
                'dataset': str(source.resolve()), 'split': 'val_unseen',
                'dev_indices': [e['index'] for e in previous['episodes']],
                'dev_scenes': sorted(dev_scenes), 'evaluation_scenes': scenes,
                'selection': chosen, 'runs_per_route': 4,
                'acceptance': 'combined_success_count_increases_and_each_backend_does_not_decrease',
                'training_use': 'forbidden', 'tuning_round_limit': 3}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'routes': len(chosen), 'scenes': len(scenes)}, indent=2))


if __name__ == '__main__':
    main()
