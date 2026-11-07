"""Resumable, bounded paired evaluation. No held-out tuning or result-based resampling."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'navgpt').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def infrastructure_error(error):
    message=str(error or '').lower()
    # A policy that consumes its episode deadline has failed that trial. It is
    # not entitled to a second rollout under the network-retry policy.
    if 'episode_timeout' in message or 'runner timeout' in message:
        return False
    return bool(message and any(word in message for word in (
        '429','rate_limit','connection','timeout','timed out','out of memory',
        '503','502','service unavailable')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--lanes',required=True,help='List of backend/config-file lane definitions')
    parser.add_argument('--stage',choices=('dev','eval'),required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.stage=='dev' and not 1<=args.round<=3: raise ValueError('Only three dev rounds permitted')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    # An output directory belongs to one controller. Never let a resumed controller
    # overlap the original and issue commands to the same simulator sessions.
    with (out/'controller.lock').open('a') as controller_lock:
        try:
            fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another evaluation controller owns this output directory')
        run(args,out)


def run(args,out):
    manifest=load(args.manifest);lanes=load(args.lanes)
    lane_configs={lane['config']:load(ROOT/lane['config']) for lane in lanes}
    fingerprint=code_hash()
    protocol={'stage':args.stage,'round':args.round,'code_sha256':fingerprint,
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'lanes':lanes,'lane_configs':lane_configs,
              'primary_success':'environment_success AND explicit_stop AND no_run_error AND no_abort',
              'notes':'One run per condition; only infrastructure failures get one retry. No best-of selection.'}
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and load(protocol_path)!=protocol:
        raise ValueError('Resume protocol differs; use the same frozen code and settings')
    write(protocol_path,protocol)
    indices=([e['index'] for e in manifest['selection']] if args.stage=='eval' else manifest['dev_indices'])
    results={};lock=threading.RLock()
    results_path=out/'results.json'
    if results_path.exists():results=load(results_path)['runs']

    def report():
        stats={}
        for row in results.values():
            key=row['backend']+'_'+row['mode'];g=stats.setdefault(key,{'count':0,'successes':0,'errors':0})
            g['count']+=1;g['successes']+=row['task_success'];g['errors']+=bool(row['error'])
        write(results_path,{'protocol':protocol,'runs':results,'aggregate':stats})
        print(json.dumps({'finished':len(results),'aggregate':stats}),flush=True)

    def lane_run(lane):
        backend=lane['backend'];base=lane_configs[lane['config']]
        lane_indices=indices[lane.get('shard',0)::lane.get('shards',1)]
        for ordinal,index in enumerate(lane_indices):
            modes=(['global','subtask'] if ordinal%2==0 else ['subtask','global']) if args.stage=='eval' else ['subtask']
            for mode in modes:
                key=f'{backend}_{mode}_{index}'
                with lock:
                    if key in results:continue
                if code_hash()!=fingerprint:raise RuntimeError('Source changed during frozen run')
                cfg=json.loads(json.dumps(base));cfg['name']=key;cfg['dataset']['episodes']=str(index)
                cfg['harness']['mode']=mode;cfg['planner'].pop('tools',None)
                cfg['output']['dir']=str(out/'outputs')
                cfg['planner']['max_turns']=100;cfg['planner']['episode_timeout_s']=1200
                config=out/'configs'/(key+'.json');write(config,cfg)
                final=None
                job_started=time.monotonic()
                attempts=[]
                for attempt in range(2):
                    run_cfg=cfg.copy();run_cfg['name']=key+('_retry1' if attempt else '')
                    write(config,run_cfg)
                    run_root=out/'outputs'/run_cfg['name']
                    log=out/(run_cfg['name']+'.log')
                    started=time.monotonic()
                    summary_path=run_root/'summary.json'
                    complete_summary=summary_path.exists() and bool(load(summary_path).get('episodes'))
                    process_file=out/(run_cfg['name']+'.process.json')
                    if process_file.exists():
                        process=load(process_file)
                        pid=process['pid']
                        # The start time prevents PID reuse from being mistaken for
                        # a child of an interrupted controller.
                        stat=Path(f'/proc/{pid}/stat')
                        if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==process['start_ticks']:
                            raise RuntimeError(f'{key} still has a live runner ({pid}); refusing duplicate execution')
                    if run_root.exists() and not complete_summary:
                        raise RuntimeError(f'Incomplete previous attempt at {run_root}; inspect it before resuming')
                    try:
                        if not complete_summary:
                            with log.open('w') as stream:
                                proc=subprocess.Popen([sys.executable,'-u','-m','navgpt','run','--config',str(config),
                                      '--paths',str(ROOT/'reproduction/lightnav_harness_20261106/local_paths.json')],
                                      cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                                stat=Path(f'/proc/{proc.pid}/stat').read_text()
                                write(process_file,{'pid':proc.pid,'start_ticks':stat.rsplit(')',1)[1].split()[19],
                                                    'started_unix':time.time()})
                                try:
                                    proc.wait(timeout=1350)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid, signal.SIGTERM)
                                    try:proc.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(proc.pid, signal.SIGKILL);proc.wait()
                                    raise
                                finally:
                                    process=load(process_file)
                                    process['wall_s']=round(time.monotonic()-started,2)
                                    process['returncode']=proc.returncode
                                    write(process_file,process)
                        episodes=load(summary_path).get('episodes',[]) if summary_path.exists() else []
                        episode=episodes[0] if episodes else None
                        error=(episode.get('error') or episode.get('api_error')) if episode else (
                            'runner exited without summary: '+(log.read_text()[-2000:] if log.exists() else 'no log'))
                        exit_code=load(process_file).get('returncode') if process_file.exists() else None
                        if exit_code not in (None,0):
                            error=error or f'runner process exit {exit_code}'
                    except subprocess.TimeoutExpired:
                        episode=None;error='runner timeout'
                    state_path=run_root/'live'/f'ep{index:04d}'/'subtasks'/'state.json'
                    state=load(state_path) if state_path.exists() else None
                    metrics=(episode or {}).get('metrics') or {}
                    backend_error=None
                    events=state_path.parent/'events.jsonl'
                    if events.exists():
                        for line in events.read_text().splitlines():
                            event=json.loads(line)
                            if event.get('kind')=='backend_error':backend_error=event['error']
                    error=error or backend_error
                    success=bool(metrics.get('success') and metrics.get('stop_called') and not error
                                 and not (state or {}).get('aborted'))
                    if mode=='subtask':
                        success=success and bool(state and state['tasks']
                                   and all(t['status']=='accepted' for t in state['tasks']))
                    final={'backend':backend,'mode':mode,'index':index,'task_success':int(success),
                           'metrics':metrics,'agent':(episode or {}).get('agent'), 'error':error,
                           'subtask_state':state,'attempt':attempt,'summary':str(run_root/'summary.json'),
                           'log':str(log),'wall_s':(load(process_file).get('wall_s') if process_file.exists() else None)
                                                  or (episode or {}).get('wall_s') or round(time.monotonic()-started,2)}
                    attempts.append({'attempt':attempt,'error':error,'log':str(log),
                                     'wall_s':final['wall_s']})
                    expected=next((e for e in manifest['selection'] if e['index']==index),None)
                    if expected and episode:
                        for field in ('episode_id','scene_id','instruction'):
                            if str(episode[field])!=str(expected[field]):raise RuntimeError('Frozen sample mismatch: '+field)
                    if not infrastructure_error(error) or attempt:break
                    time.sleep(60 if any(w in str(error).lower() for w in ('429','rate_limit')) else 10)
                with lock:
                    final['attempts']=attempts
                    final['wall_s']=round(max(time.monotonic()-job_started,sum(a['wall_s'] for a in attempts)),2)
                    results[key]=final;report()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures=[executor.submit(lane_run,lane) for lane in lanes]
        for future in concurrent.futures.as_completed(futures):future.result()
    with lock:report()


if __name__=='__main__':main()
