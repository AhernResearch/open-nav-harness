"""Exercise evaluation recovery without simulator services or model calls."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_runner', ROOT/'reproduction/run_subtask_paired.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

STUB = '''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--config');p.add_argument('--paths')
a=p.parse_args();c=json.loads(pathlib.Path(a.config).read_text());root=pathlib.Path(c['output']['dir'])/c['name']
root.mkdir(parents=True)
with (root.parent/'model_calls.txt').open('a') as f:f.write(c['name']+'\\n')
e={'episode_id':'0','scene_id':'fixture','instruction':'fixture','metrics':{'success':1,'stop_called':1}}
(root/'summary.json').write_text(json.dumps({'episodes':[e]}))
if c['harness']['mode']=='subtask':
 s=root/'live/ep0000/subtasks';s.mkdir(parents=True)
 (s/'state.json').write_text(json.dumps({'tasks':[{'status':'accepted'}]}))
'''


class RunnerRecoveryTests(unittest.TestCase):
    def test_policy_deadlines_cannot_receive_network_retries(self):
        self.assertFalse(runner.infrastructure_error('episode_timeout after 1200s'))
        self.assertFalse(runner.infrastructure_error('runner timeout'))
        self.assertTrue(runner.infrastructure_error('ReadTimeout: connection timed out'))
        self.assertTrue(runner.infrastructure_error('HTTP 429 rate_limit'))

    def test_resume_recovers_completed_summary_without_a_second_model_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'navgpt').mkdir()
            (root/'navgpt/__main__.py').write_text(STUB)
            (root/'config.json').write_text(json.dumps({'harness':{},'planner':{},'dataset':{},'output':{}}))
            manifest={'selection':[{'index':0,'episode_id':'0','scene_id':'fixture','instruction':'fixture'}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'lanes.json').write_text(json.dumps([{'backend':'lightnav','config':'config.json'}]))
            args=['runner','--stage','eval','--manifest',str(root/'manifest.json'),
                  '--lanes',str(root/'lanes.json'),'--output',str(root/'run')]
            def invoke():
                with patch.object(runner,'ROOT',root),patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
            invoke()
            results=root/'run/results.json';data=json.loads(results.read_text())
            del data['runs']['lightnav_global_0'];results.write_text(json.dumps(data))
            invoke()
            self.assertEqual(len((root/'run/outputs/model_calls.txt').read_text().splitlines()),2)
            self.assertEqual(len(json.loads(results.read_text())['runs']),2)

            # A CLI writes an empty initial summary before inference begins. It
            # must never be confused with a completed run on controller resume.
            del data['runs']['lightnav_subtask_0'];results.write_text(json.dumps(data))
            (root/'run/outputs/lightnav_subtask_0/summary.json').write_text('{"episodes":[]}')
            with self.assertRaisesRegex(RuntimeError,'Incomplete previous attempt'):
                invoke()
            self.assertEqual(len((root/'run/outputs/model_calls.txt').read_text().splitlines()),2)


if __name__=='__main__':unittest.main()
