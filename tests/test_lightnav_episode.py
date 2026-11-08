"""Regression checks for the episode timeline, independent of CUDA/model weights."""
import pytest

from navgpt.planner.prompts import build_briefing_for
from navgpt.planner.tools.session import NavigationSession
from navgpt.planner.tools.vla_client import NavGPTVLAClient
from navgpt.vla.lightnav import LightNavBridge


class RecordingInference:
    def __init__(self):
        self.frames, self.predictions, self.resets = [], [], 0

    def __call__(self, operation, data):
        if operation == "reset":
            self.resets += 1
            self.frames = []
            return {"rc": 0}
        if operation == "next":
            self.frames.append(data["image"])
        if data.get("instruction"):
            self.predictions.append((list(self.frames), data["instruction"]))
        return {"rc": 0, "stop": False, "actions": {"actions": [[.1, 0., 0.]] * 10}}


def connection():
    model = RecordingInference()
    bridge = LightNavBridge(inference=model)

    def transport(method, path, data=None):
        if method == "GET":
            return {"/info": bridge.info, "/state": bridge.state}[path]()
        return {"/observe": bridge.observe, "/act": bridge.act,
                "/reset": bridge.reset, "/context": bridge.replace_context}[path](data)

    client = NavGPTVLAClient("", backend="lightnav", transport=transport)
    client.reset()
    return model, bridge, client


def test_long_episode_preserves_timeline_across_tasks_repairs_and_repeated_endpoints():
    model, bridge, client = connection()
    frames = []
    for i in range(120):
        frames.append({"id": f"frame-{i}", "views": {"policy_front": f"image-{i}"}})
        # Some images come from retracing: append history without generating.
        if 40 <= i < 50:
            continue
        client.prepare_context(f"task-{i // 30}", frames)
        client.act(frames[-1]["views"], "Go through the door", "episode", None, None,
                   frame_id=frames[-1]["id"])
        assert model.frames == [f"image-{j}" for j in range(i + 1)]
    client.prepare_context("another-task", frames)
    client.act(frames[-1]["views"], "Turn left", "episode", None, None,
               frame_id=frames[-1]["id"])
    assert model.resets == 1
    assert len(model.frames) == bridge.state()["history_frames"] == 120
    assert model.predictions[-1] == ([f"image-{j}" for j in range(120)], "Turn left")
    with pytest.raises(ValueError, match="extend"):
        client.prepare_context("task", frames[-16:])
    with pytest.raises(ValueError, match="rewrite"):
        bridge.act({"frame_id": "frame-119", "rgb_policy": "changed", "text": "Go"})
    with pytest.raises(ValueError, match="reorder"):
        bridge.observe({"frame_id": "frame-2", "rgb_policy": "image-2"})
    client.reset()
    assert bridge.state()["history_frames"] == 0 and model.frames == []


def test_lightnav_task_instruction_has_no_map_caption_and_no_manual_reset_tool():
    session = NavigationSession(env={"NAVGPT_CONDITION": "planner_agentic",
                                     "NAVGPT_VLA_BACKEND": "lightnav"})
    m = session._tasks
    m.observe({"policy_front": "Zg=="}, {}, 0)
    m.update_plan([{"id": 1, "instruction": "Cross the door", "status": "pending"}], "Initial")
    m.begin(1)
    m.memory.begin({"position": [0, 0, 0]}, "f0", "execute_subtask")
    m.memory.finish({"position": [0, 0, 0]}, "f1", "vla_stop")
    m.remember(0, 0, "We may have entered the wrong room.", {"map_call": 1})
    assert m.policy_instruction(session._input_profile.include_memory) == "Cross the door"
    assert "wrong room" in m.policy_instruction()  # Native actor contract unchanged.
    definition = session.tools.definitions["execute_subtask"]
    assert set(definition.inputSchema["properties"]) == {"task_id", "instruction"}
    assert "runtime-managed" in definition.description
    assert "reset_context" not in definition.description
    briefing = build_briefing_for("planner_agentic", "Go", 500, actor_history="episode")
    assert "reset_context" not in briefing and "runtime-managed" in briefing
    with pytest.raises(ValueError, match="unsupported"):
        session.execute_agentic_task(1, reset_context=True)


def test_official_vla_only_budget_stop_is_not_reported_as_model_prediction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from navgpt.planner import vla_only
    from navgpt.planner.tools import vla_client
    from navgpt.vla.profiles import input_profile
    from unittest.mock import Mock
    client = SimpleNamespace(input_profile=input_profile('lightnav'), act=Mock())
    monkeypatch.setattr(vla_client, 'NavGPTVLAClient', lambda *a, **kw: client)
    for name in ['_reset', 'panel_field', 'panel_action']:
        monkeypatch.setattr(vla_only, name, lambda *a, **kw: None)
    calls = []
    def call(url, method, data):
        calls.append((method, data))
        return {'instruction': 'Go to the door', 'metrics': {}}
    monkeypatch.setattr(vla_only, 'call_function', call)
    cfg = SimpleNamespace(env_url='local', verb_prefix='r2rce', vla_url='local',
                          vla_backend='lightnav', dataset='r2r', step_budget=1, episode_timeout=60)
    result = vla_only.run_episode(cfg, 0, tmp_path)
    client.act.assert_not_called()
    assert result['agent']['forced_stop'] and result['agent']['called_stop']
    assert ('r2rce__step_discrete', {'action': 0}) in calls


@pytest.mark.parametrize('policy,expected_resets', [('episode', 1), ('completed_subtask', 2)])
def test_completed_task_crop_retains_contiguous_retrace_and_never_crops_on_return(policy, expected_resets):
    model, bridge, client = connection()
    session = NavigationSession(env={
        'NAVGPT_CONDITION': 'planner_agentic', 'NAVGPT_VLA_BACKEND': 'lightnav',
        'NAVGPT_ACTOR_HISTORY_POLICY': policy,
    })
    session._vla = client
    m = session._tasks
    def observe(n):
        m.mark_motion()
        m.observe({'policy_front': f'image-{n}'}, {}, n)
    def predict(instruction):
        frames = m.frames[session._actor_history_start:]
        client.prepare_context(m.context_key, frames)
        client.act(frames[-1]['views'], instruction, 'ep', None, None, frame_id=frames[-1]['id'])
        return []
    session.vla_navigate = predict
    plan = [{'id': 1, 'instruction': 'Exit', 'status': 'pending'},
            {'id': 2, 'instruction': 'Turn left', 'status': 'pending'}]
    observe(0)
    m.update_plan(plan, 'Initial')
    session.execute_agentic_task(1)
    for i in range(1, 21):
        observe(i)
    session.execute_agentic_task(1, 'Keep going')
    assert model.resets == 1 and len(model.frames) == 21
    # A task switch without completion must also keep history.
    session.execute_agentic_task(2)
    session.execute_agentic_task(1)
    assert model.resets == 1
    plan[0]['status'] = 'completed'
    m.update_plan(plan, 'Door crossed')
    m.update_plan(plan, 'Same completion, clarified next task')
    assert model.resets == 1
    for i in range(21, 24):  # Passive movement between completion and next execution.
        observe(i)
    session.execute_agentic_task(2)
    expected = list(range(20 if policy == 'completed_subtask' else 0, 24))
    assert model.frames == [f'image-{i}' for i in expected]
    assert model.resets == expected_resets
    session.execute_agentic_task(2, 'Continue')
    assert model.resets == expected_resets
    assert bridge.state()['history_frames'] == len(expected)
    assert len(m.frames) == 24  # Planner evidence is never deleted.
    # A new completion followed by reopening must cancel the pending crop.
    plan[1]['status'] = 'completed'
    m.update_plan(plan, 'Claim arrival')
    plan[1]['status'] = 'pending'
    plan[0]['status'] = 'pending'
    m.update_plan(plan, 'Correction: wrong room')
    assert m.completed_boundary is None
    session.execute_agentic_task(1)
    assert model.resets == expected_resets


def test_history_policy_configuration_is_explicit_and_scoped():
    from pathlib import Path
    from navgpt.settings import Settings, SettingsError
    data = {'planner': {'runtime': 'codex'}, 'vla': {'backend': 'lightnav'},
            'harness': {'mode': 'agentic', 'actor_history_policy': 'completed_subtask'}}
    assert Settings(Path('test.json'), data, {}).run_config().actor_history_policy == 'completed_subtask'
    for policy in ['typo', None, True]:
        data['harness']['actor_history_policy'] = policy
        with pytest.raises(SettingsError):
            Settings(Path('test.json'), data, {}).run_config()
    data['harness']['actor_history_policy'] = 'completed_subtask'
    data['vla']['backend'] = 'navgpt'
    with pytest.raises(SettingsError):
        Settings(Path('test.json'), data, {}).run_config()
