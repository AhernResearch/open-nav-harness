import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path

from navgpt.planner.tools.subtasks import SubtaskManager
from navgpt.settings import load, SUBTASK_TOOLS
from navgpt.vla.context import AgentContext, context_frames
from navgpt.vla.lightnav import LightNavBridge
from test_lightnav import FakeWS


def task(ident=1, quote='Exit the room', instruction='Exit the room'):
    return {'id': ident, 'source_quote': quote, 'instruction': instruction,
            'completion_criterion': 'Robot is through the doorway'}


class TaskTests(unittest.TestCase):
    def manager(self):
        m = SubtaskManager('Exit the room then turn left')
        m.observe({'policy_front': 'Zg=='}, {}, 0)
        return m

    def review(self, m, outcome='accepted'):
        m.mark_map()
        return m.review(outcome, [m.frames[-1]['id'], m.map_ref], 'Exited room.', 'Door crossed.')

    def test_plan_and_completion_gates(self):
        m = self.manager()
        with self.assertRaises(ValueError): m.begin()
        m.update_plan([task(), task(2, 'turn left', 'Turn left')], 'Local transitions')
        m.begin(); m.mark_motion()
        with self.assertRaises(ValueError): m.begin()
        with self.assertRaises(ValueError): m.finish('complete')
        with self.assertRaises(ValueError): self.review(m)
        m.observe({'policy_front': 'Zw=='}, {}, 1)
        self.review(m)
        self.assertEqual(m.begin()['id'], 2)
        self.review(m)
        m.finish('complete')
        self.assertTrue(all(t['status'] == 'accepted' for t in m.tasks))

    def test_budget_stop_does_not_advance_and_replan_requires_update(self):
        m = self.manager(); m.update_plan([task()], 'Initial')
        m.begin(); m.returned('distance_budget')
        self.assertEqual(m.current()['id'], 1)
        self.review(m, 'continue'); m.begin()
        self.review(m, 'replan')
        with self.assertRaises(ValueError): m.begin()
        m.update_plan([task(instruction='Pass through the visible room exit')], 'Clarified exit')
        self.assertEqual(m.begin()['id'], 1)

    def test_evidence_and_source_quotes(self):
        m = self.manager()
        with self.assertRaises(ValueError): m.update_plan([task(quote='invented room')], 'Initial')
        m.update_plan([task()], 'Initial'); m.begin(); m.mark_map()
        with self.assertRaises(ValueError): m.review('accepted', [m.map_ref], 'done', 'done')
        with self.assertRaises(ValueError): m.review('accepted', [m.frames[-1]['id'], m.map_ref], 'x'*801, 'done')
        m.mark_motion()
        with self.assertRaises(ValueError): self.review(m)

    def test_accepted_tasks_cannot_be_silently_changed(self):
        m = self.manager(); m.update_plan([task()], 'Initial');m.begin();self.review(m)
        with self.assertRaises(ValueError): m.update_plan([task(instruction='Return')], 'Change')
        m.update_plan([task(instruction='Pass through exit')], 'Prior evidence contradicted', [1])
        self.assertEqual(m.current()['status'], 'pending')

    def test_context_has_two_boundaries_anchor_and_recent_frames(self):
        m = self.manager(); m.update_plan([task()], 'Initial'); m.begin()
        for i in range(1, 25): m.observe({'policy_front': base64.b64encode(str(i).encode()).decode()}, {}, i)
        self.review(m, 'continue')
        m.update_plan([task()], 'Restart from corrected position',reopen_ids=[1]); m.begin()
        anchor = m.anchor
        for i in range(25, 65): m.observe({'policy_front': base64.b64encode(str(i).encode()).decode()}, {}, i)
        chosen = m.selected_frames()
        self.assertEqual(len(chosen), 16)
        self.assertEqual([f['id'] for f in chosen[:3]], [m.frames[i]['id'] for i in (anchor-2, anchor-1, anchor)])
        self.assertEqual(chosen[-1]['id'], m.frames[-1]['id'])
        self.assertEqual(len({f['id'] for f in chosen}), 16)

    def test_residual_command_revision_retains_same_subtask_anchor(self):
        m=self.manager();m.update_plan([task()], 'Initial');m.begin()
        anchor=m.anchor
        for i in range(1,8):
            m.observe({'policy_front':base64.b64encode(str(i).encode()).decode()}, {}, i)
        self.review(m,'replan')
        m.update_plan([task(instruction='Cross the exit threshold')], 'Approach already completed')
        m.begin()
        self.assertEqual(m.anchor,anchor)
        self.assertEqual(m.selected_frames()[0]['id'],m.frames[anchor]['id'])
        self.assertEqual(len(m.selected_frames()),8)
        self.review(m,'replan')
        m.update_plan([task()], 'Explicitly restart the same goal',reopen_ids=[1]);m.begin()
        self.assertEqual(m.anchor,len(m.frames)-1)

    def test_archive_dedup_and_abort(self):
        with tempfile.TemporaryDirectory() as folder:
            m = SubtaskManager('Exit the room', folder)
            a = m.observe({'policy_front': 'Zg=='}, {}, 0)
            self.assertEqual(a, m.observe({'policy_front': 'Zg=='}, {}, 0))
            self.assertEqual((Path(folder)/'observations'/a/'policy_front.png').read_bytes(), b'f')
            m.finish('abort')
            self.assertTrue(json.loads((Path(folder)/'state.json').read_text())['aborted'])

    def test_revised_local_instruction_precedes_history_and_excludes_pending_route(self):
        m = self.manager()
        m.update_plan([task(), task(2, 'turn left', 'Turn left')], 'Initial')
        m.begin();m.mark_map()
        m.review('replan',[m.frames[-1]['id'],m.map_ref],
                 'Reached the exit threshold but have not crossed it.', 'Only the crossing remains')
        m.update_plan([task(instruction='Cross the doorway threshold'),
                       task(2, 'turn left', 'Turn left')], 'Remove the completed approach')
        m.begin();text=m.policy_instruction()
        self.assertTrue(text.startswith('Cross the doorway threshold'))
        self.assertIn('Reached the exit threshold',text)
        self.assertNotIn('Turn left',text)
        self.assertNotIn('turn left',text)


class ContextTests(unittest.TestCase):
    def test_lightnav_refuses_lossy_context_replacement(self):
        bridge = LightNavBridge('ws://local', connect=lambda *a, **k: FakeWS())
        with self.assertRaisesRegex(ValueError, 'append-only episode history'):
            bridge.replace_context({'context_key': 'task', 'frames': []})

    def test_navgpt_reuses_only_matching_frame_cache_indices(self):
        class Agent:
            rgb_lists = [[] for _ in range(4)]
            def replace_history(self, groups, reuse):
                self.reuse = reuse
                self.rgb_lists = [[g[c] for g in groups] for c in range(4)]
        agent=Agent();ctx=AgentContext(agent, lambda x:x)
        def frame(i): return {'id':str(i),'views':{c:str(i)+c for c in ctx.cameras}}
        ctx.replace({'context_key':'1','frames':[frame(0),frame(1)]})
        self.assertEqual(agent.reuse,[None,None])
        ctx.replace({'context_key':'2','frames':[frame(1),frame(2)]})
        self.assertEqual(agent.reuse,[1,None])
        ctx.replace({'context_key':'2','frames':[{'id':'1'},{'id':'2'}]})
        self.assertEqual(agent.reuse,[0,1])
        ctx.validate_act({'context_key':'2','frame_id':'3'})
        with self.assertRaises(ValueError): ctx.validate_act({'context_key':'old','frame_id':'3'})

    def test_configuration_preserves_global_default(self):
        base = {'planner':{'runtime':'codex','model':'gpt-5.6-sol'}}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'experiment.json';p.write_text(json.dumps(base))
            self.assertEqual(load(p).run_config().condition,'planner_vla_memory')
            base['harness']={'mode':'subtask'};p.write_text(json.dumps(base))
            cfg=load(p).run_config()
            self.assertEqual(cfg.condition,'planner_subtask')
            self.assertEqual(set(cfg.tools),set(SUBTASK_TOOLS))
            self.assertTrue(cfg.vla_map_enabled())
            base['harness']['subtask_max_actions']=0;p.write_text(json.dumps(base))
            with self.assertRaises(ValueError):load(p).run_config()

    def test_subtask_configuration_cannot_silently_run_without_a_planner(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'experiment.json'
            p.write_text(json.dumps({'harness':{'mode':'subtask'}}))
            with self.assertRaisesRegex(ValueError,'requires a Planner'):
                load(p).run_config()


if __name__ == '__main__': unittest.main()
