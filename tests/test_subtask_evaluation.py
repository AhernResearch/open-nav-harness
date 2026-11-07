import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PairedAcceptanceTests(unittest.TestCase):
    def evaluate(self, folder, light=(20, 25), nav=(20, 20), broken_history=False,
                 changed_manifest=False, retained_anchor=False):
        folder=Path(folder)
        events=folder/'outputs/live/ep0000/subtasks/events.jsonl'
        events.parent.mkdir(parents=True,exist_ok=True)
        rows=[{'kind':'observation','frame_id':'frame-000000'},
              {'kind':'begin','context_key':'1:1'},
              {'kind':'policy_context','context_key':'1:1','event':3,
               'frame_ids':['frame-000000']*(2 if broken_history else 1)}]
        if retained_anchor:
            rows.extend({'kind':'observation','frame_id':f'frame-{i:06d}'} for i in range(1,21))
            rows.extend([{'kind':'begin','context_key':'2:1','anchor_frame_id':'frame-000000'},
                         {'kind':'policy_context','context_key':'2:1','event':25,
                          'frame_ids':['frame-000000']+[f'frame-{i:06d}' for i in range(7,21)]}])
        events.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        runs={}
        for backend,counts in [('lightnav',light),('navgpt',nav)]:
            for mode,count in zip(('global','subtask'),counts):
                for index in range(50):
                    runs[f'{backend}_{mode}_{index}']={
                        'task_success':int(index<count),'metrics':{'distance_to_goal':1.0},
                        'error':None,'wall_s':1,'summary':str(folder/'outputs/summary.json')}
        result=folder/'results.json';manifest=folder/'manifest.json'
        manifest.write_text(json.dumps({'selection':[{'index':i} for i in range(50)]}))
        fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest()
        result.write_text(json.dumps({'runs':runs,'protocol':{'manifest_sha256':fingerprint}}))
        if changed_manifest:
            manifest.write_text(json.dumps({'selection':[{'index':i} for i in range(50)],'seed':999}))
        subprocess.run([sys.executable,str(ROOT/'reproduction/analyze_subtask_paired.py'),
                        '--results',str(result),'--manifest',str(manifest)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        return json.loads((folder/'comparison_report.json').read_text())

    def test_combined_gain_and_no_backend_regression_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.evaluate(folder)
            self.assertTrue(report['acceptance_passed'])
            self.assertEqual(report['combined_sr_delta_percentage_points'],5.0)
            self.assertTrue(report['history_audit_passed'])

    def test_positive_combined_gain_cannot_hide_backend_regression(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.evaluate(folder,light=(20,30),nav=(20,19))
            self.assertGreater(report['combined_sr_delta_percentage_points'],0)
            self.assertFalse(report['acceptance_passed'])

    def test_duplicate_image_context_cannot_pass_even_with_success_gain(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.evaluate(folder,broken_history=True)
            self.assertFalse(report['history_audit_passed'])
            self.assertFalse(report['acceptance_passed'])

    def test_changed_frozen_manifest_cannot_be_used_to_score_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(subprocess.CalledProcessError):
                self.evaluate(folder,changed_manifest=True)

    def test_revised_same_goal_history_uses_recorded_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.evaluate(folder,retained_anchor=True)
            self.assertTrue(report['history_audit_passed'])


if __name__=='__main__':unittest.main()
