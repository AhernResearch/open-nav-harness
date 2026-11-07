"""Analyze every frozen route, with route-clustered paired bootstrap intervals."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def history_audit(summary):
    folder=Path(summary).parent/'live'
    contexts=0;max_frames=0;max_old=0;violations=[]
    for path in folder.glob('*/subtasks/events.jsonl'):
        latest=-1;anchors={}
        for line in path.read_text().splitlines():
            event=json.loads(line);kind=event['kind']
            if kind=='observation':latest=int(event['frame_id'].split('-')[1])
            if kind=='begin':anchors.setdefault(event['context_key'],
                int(event['anchor_frame_id'].split('-')[1]) if event.get('anchor_frame_id') else latest)
            if kind!='policy_context':continue
            contexts+=1
            frames=event['frame_ids'];numbers=[int(i.split('-')[1]) for i in frames]
            old=sum(n<anchors[event['context_key']] for n in numbers)
            max_frames=max(max_frames,len(frames));max_old=max(max_old,old)
            if (len(frames)>16 or len(frames)!=len(set(frames)) or numbers!=sorted(numbers)
                    or old>2 or not numbers or anchors[event['context_key']] not in numbers
                    or max(numbers)>latest):
                violations.append({'event':event['event'],'context_key':event['context_key']})
    return {'contexts':contexts,'max_frame_groups':max_frames,'max_previous_task_groups':max_old,
            'violations':violations}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True)
    parser.add_argument('--manifest',required=True);args=parser.parse_args()
    results=json.loads(Path(args.results).read_text());manifest=json.loads(Path(args.manifest).read_text())
    fingerprint=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    expected_fingerprint=results.get('protocol',{}).get('manifest_sha256')
    if expected_fingerprint and expected_fingerprint!=fingerprint:
        raise ValueError('Scoring manifest differs from the frozen run protocol')
    runs=results['runs'];indices=[e['index'] for e in manifest['selection']]
    if not indices or len(set(indices))!=len(indices):
        raise ValueError('Scoring requires distinct frozen episode indices')
    missing=[];aggregates={};audits={};pairs=[]
    for backend in ('lightnav','navgpt'):
        for mode in ('global','subtask'):
            selected=[]
            for index in indices:
                key=f'{backend}_{mode}_{index}'
                if key not in runs:missing.append(key);continue
                row=runs[key];selected.append(row)
                if mode=='subtask':audits[key]=history_audit(row['summary'])
            metrics={}
            for metric in ('spl','distance_to_goal','oracle_success','ndtw','path_length','steps_taken'):
                values=[r['metrics'][metric] for r in selected if metric in r['metrics']]
                metrics[metric]={'mean':sum(values)/len(values) if values else None,'count':len(values)}
            calls=collections.Counter();usage=collections.Counter()
            for row in selected:
                agent=row.get('agent') or {}
                calls.update(agent.get('tool_calls') or {})
                usage.update({k:v for k,v in (agent.get('usage') or {}).items()
                              if isinstance(v,(int,float))})
            aggregates[backend+'_'+mode]={'count':len(selected),'successes':sum(r['task_success'] for r in selected),
               'sr':sum(r['task_success'] for r in selected)/len(indices),
               'errors':sum(bool(r['error']) for r in selected),'metrics':metrics,
               'environment_successes':sum(bool(r['metrics'].get('success')) for r in selected),
               'tool_calls_total':dict(calls),'planner_usage_total':dict(usage),
               'mean_tool_calls':sum(calls.values())/len(selected) if selected else None,
               'wall_s_total':sum(r['wall_s'] for r in selected)}
    complete=not missing
    delta=[]
    for index in indices:
        row={'index':index}
        for backend in ('lightnav','navgpt'):
            a=runs.get(f'{backend}_global_{index}');b=runs.get(f'{backend}_subtask_{index}')
            row[backend]={'global':a['task_success'] if a else None,
                          'subtask':b['task_success'] if b else None}
        pairs.append(row)
        if all(row[b]['global'] is not None and row[b]['subtask'] is not None for b in ('lightnav','navgpt')):
            delta.append(sum(row[b]['subtask']-row[b]['global'] for b in ('lightnav','navgpt'))/2)
    interval=None
    if complete:
        rng=random.Random(20261107)
        samples=sorted(100*sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(10000))
        interval=[samples[249],samples[9749]]
    gains={b:aggregates[b+'_subtask']['successes']-aggregates[b+'_global']['successes']
           for b in ('lightnav','navgpt')}
    transitions={}
    for backend in ('lightnav','navgpt'):
        transitions[backend]={}
        for label,before,after in [('rescued',0,1),('regressed',1,0),('both_success',1,1),('both_failure',0,0)]:
            transitions[backend][label]=[p['index'] for p in pairs
                if p[backend]=={'global':before,'subtask':after}]
    audit_ok=complete and all(a['contexts']>0 and not a['violations'] for a in audits.values())
    passed=bool(complete and audit_ok and sum(gains.values())>0 and all(g>=0 for g in gains.values()))
    report={'complete':complete,'missing':missing,'acceptance_passed':passed,
            'paired_success_gains':gains,'paired_transitions':transitions,'aggregate':aggregates,'pairs':pairs,
            'combined_sr_delta_percentage_points':100*sum(delta)/len(delta) if delta else None,
            'route_clustered_bootstrap_95_percent_ci':interval,
            'history_audit':audits,'history_audit_passed':audit_ok,'protocol':results['protocol'],
            'interpretation':'Observed paired improvement gate; interval is reported, not used as a significance gate.'}
    path=Path(args.results).with_name('comparison_report.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('complete','acceptance_passed','paired_success_gains',
                   'combined_sr_delta_percentage_points','route_clustered_bootstrap_95_percent_ci','history_audit_passed')},indent=2))
    print(path)


if __name__=='__main__':main()
