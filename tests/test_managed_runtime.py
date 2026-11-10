import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from navgpt.planner.tools.registry import ToolResult as CallToolResult, TextContent, ImageContent
from open_nav_harness.core.config import PlannerProfile
from open_nav_harness.providers import runtime as module
from open_nav_harness.providers.runtime import ManagedRuntime, tool_feedback
from navgpt.planner.contracts import EpisodeContext
from navgpt.planner.runner import EventSink


def call(ident, name, arguments="{}"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": arguments}}


async def run_fake(tmp_path, monkeypatch, responses, names=None, results=None, schemas=None, sink_state=None):
    names = names or ["observe_panorama", "terminate_episode"]
    executed = []

    class Client:
        def __init__(self, _, journal=None):
            pass

        async def count(self, *_):
            return 200

        async def close(self):
            pass

        async def generate(self, *_):
            return {"choices": [{"message": {"role": "assistant", "tool_calls": responses.pop(0)}}]}

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return [
                SimpleNamespace(
                    name=name,
                    description=name,
                    inputSchema=(schemas or {}).get(
                        name, {"type": "object", "properties": {}, "additionalProperties": False}
                    ),
                )
                for name in names
            ]

        async def call_tool(self, name, args):
            executed.append(name)
            value = {
                "steps_taken_total": 0,
                "episode_over": name == "terminate_episode",
                "end_reason": "stop_called" if name == "terminate_episode" else "",
            }
            if results and name in results:
                value = results[name]
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))])

    @asynccontextmanager
    async def transport(*_):
        yield Session()

    monkeypatch.setattr(module, "CompatibleClient", Client)
    monkeypatch.setattr(module, "open_tools", transport)
    ctx = EpisodeContext(
        "system",
        "first",
        "test",
        5,
        None,
        {},
        None,
        [],
        {"NAVGPT_CONDITION": "planner_subtask", "NAVGPT_TOOLS": json.dumps(names)},
        tmp_path,
        20,
    )
    sink = EventSink(tmp_path / "events.jsonl")
    sink.last_step_result = sink_state
    outcome = await ManagedRuntime(PlannerProfile(model="test")).run(ctx, sink)
    sink.close()
    return executed, outcome


async def test_multi_call_response_counts_once_and_stops(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch, [[call("1", "observe_panorama"), call("2", "terminate_episode")]]
    )
    assert executed == ["observe_panorama", "terminate_episode"]
    assert outcome.error is None
    assert outcome.turns == 1


async def test_explicit_ablation_contract_reaches_model_loop(tmp_path, monkeypatch):
    from navgpt.settings import AGENTIC_TOOLS

    names = [name for name in AGENTIC_TOOLS if name != "navigate_relative"]
    executed, outcome = await run_fake(
        tmp_path, monkeypatch,
        [[call("1", "navigate_relative")], [call("2", "terminate_episode")]],
        names=names,
    )
    assert outcome.error is None
    assert executed == ["terminate_episode"]
    rows = [json.loads(line) for line in (tmp_path / "trace/requests.jsonl").read_text().splitlines()]
    start = next(row for row in rows if row["kind"] == "session_start")
    assert {t["function"]["name"] for t in start["tools"]} == set(names)
    assert any(row["kind"] == "tool_validation_error" for row in rows)


async def test_invalid_batch_never_partially_executes(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path,
        monkeypatch,
        [[call("1", "observe_panorama"), call("2", "unknown")], [call("3", "terminate_episode")]],
    )
    assert executed == ["terminate_episode"]
    assert outcome.error is None
    assert outcome.turns == 2


async def test_repeated_id_never_repeats_side_effect(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch, [[call("1", "observe_panorama")], [call("1", "observe_panorama")]]
    )
    assert executed == ["observe_panorama"]
    assert "Repeated tool call id" in outcome.error


def test_multimodal_tool_feedback_preserves_call_provenance():
    feedback, images, texts = tool_feedback(
        CallToolResult(
            content=[
                TextContent(type="text", text="observation"),
                ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            ]
        ),
        "call-7",
    )
    assert feedback["tool_call_id"] == "call-7"
    assert "call-7" in images[0]["text"]
    assert images[1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
    assert feedback['content'] == ''
    assert texts == ['observation']  # Full original text still reaches the audit sink.
    assert images[0]['text'].count('observation') == 1


def test_image_label_dedup_preserves_status_and_unlabelled_images():
    status = '{"episode_over": true, "steps_taken_total": 500}'
    image = ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")
    feedback, images, texts = tool_feedback(CallToolResult(content=[
        TextContent(text="Front view"), image, image,
        TextContent(text=status), image, TextContent(text="No image follows"),
    ]), "call-8")
    assert feedback['content'] == status + '\nNo image follows'
    assert texts == ['Front view', status, 'No image follows']
    assert images[0]['text'] == 'Tool call-8: Front view'
    assert images[2]['text'] == images[4]['text'] == 'Tool call-8'
    assert sum(i['type'] == 'image_url' for i in images) == 3


async def test_caption_may_precede_but_never_follow_an_unread_map(tmp_path, monkeypatch):
    names = ["observe_map", "execute_subtask", "update_memory", "update_plan", "terminate_episode"]
    executed, outcome = await run_fake(
        tmp_path,
        monkeypatch,
        [
            # A caption after a map in the same response describes a map never read.
            [call("1", "execute_subtask"), call("2", "update_memory")],
            [call("3", "observe_map"), call("4", "update_memory")],
            # Captioning the map from the previous result, then moving on, is one turn.
            [call("5", "update_memory"), call("6", "update_plan"), call("7", "execute_subtask")],
            [call("8", "terminate_episode")],
        ],
        names=names,
    )
    assert executed == ["update_memory", "update_plan", "execute_subtask", "terminate_episode"]
    assert outcome.error is None
    events = [json.loads(line) for line in (tmp_path / "trace/requests.jsonl").read_text().splitlines()]
    rejected = [e for e in events if e["kind"] == "tool_validation_error"]
    assert len(rejected) == 2 and all("already read" in str(e) for e in rejected)


@pytest.mark.parametrize('name,status', [
    ('execute_subtask', {'steps_taken_total': 500, 'episode_over': True,
                         'end_reason': 'step_budget_exhausted', 'subtask_return': 'action_budget'}),
    ('navigate_relative', {'steps_taken_total': 500, 'episode_over': True,
                          'end_reason': 'step_budget_exhausted'}),
    ('navigate_to_node', {'step_count': 499, 'truncated': True, 'terminated': False,
                         'arrived': False, 'termination_reason': 'stopped_short'}),
])
async def test_environment_budget_ends_loop_and_skips_remaining_batch(tmp_path, monkeypatch, name, status):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch,
        [[call('move', name), call('unused', 'observe_panorama')]],
        names=[name, 'observe_panorama', 'terminate_episode'], results={name: status},
    )
    assert executed == [name]  # No more model requests, observations or STOP calls.
    assert outcome.turns == 1
    assert outcome.error is None  # A scored navigation failure, not an infrastructure error.
    assert outcome.extra['environment_end_reason'] == 'step_budget_exhausted'
    assert outcome.extra['explicit_stop'] is False
    rows = [json.loads(l) for l in (tmp_path / 'trace/requests.jsonl').read_text().splitlines()]
    terminal = next(r for r in rows if r['kind'] == 'environment_end')
    assert terminal['source_tool'] == name and terminal['steps_taken_total'] in (499, 500)
    assert rows[-1]['kind'] == 'session_end' and rows[-1]['explicit_stop'] is False


async def test_vla_stop_is_only_a_handoff_not_episode_termination(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch,
        [[call('vla', 'execute_subtask')], [call('stop', 'terminate_episode')]],
        names=['execute_subtask', 'terminate_episode'],
        results={'execute_subtask': {'steps_taken_total': 12, 'episode_over': False,
                                     'subtask_return': 'vla_stop'}},
    )
    assert executed == ['execute_subtask', 'terminate_episode']
    assert outcome.error is None and outcome.extra['explicit_stop'] is True
    assert outcome.extra['environment_end_reason'] == 'stop_called'


async def test_planner_turn_exhaustion_remains_distinct(tmp_path, monkeypatch):
    executed, outcome = await run_fake(
        tmp_path, monkeypatch, [[call(str(i), 'observe_panorama')] for i in range(5)],
    )
    assert len(executed) == 5 and outcome.error == 'planner_turn_budget_exhausted'
    assert outcome.extra['environment_end_reason'] is None
    assert outcome.extra['explicit_stop'] is False


def test_return_telemetry_keeps_last_status_and_budget_failure_is_scored():
    from navgpt.planner.runner import is_scored
    from open_nav_harness.platform.results import aggregate, metrics
    retrace = {'step_count': 499, 'truncated': True, 'terminated': False, 'arrived': False}
    parsed = EventSink.parse_step_result([json.dumps(retrace)])
    assert parsed['steps_taken_total'] == 499 and parsed['episode_over'] is True
    assert parsed['end_reason'] == 'step_budget_exhausted'
    stop = {'steps_taken_total': 500, 'episode_over': True, 'end_reason': 'stop_called'}
    assert EventSink.parse_step_result([json.dumps(retrace), json.dumps(stop)]) == stop
    result = {'agent': {'env_steps': 499, 'end_reason': parsed['end_reason'], 'called_stop': False},
              'metrics': {'success': 0, 'stop_called': 0, 'distance_to_goal': 2}, 'error': None}
    assert is_scored(result)
    row = metrics({'id': 'a', 'episode': 0, 'ordinal': 1, 'status': 'completed',
                   'result': json.dumps(result)})
    summary = aggregate([row])
    assert summary['routes_completed'] == 1 and summary['sr'] == 0
    assert summary['execution_errors'] == 0


async def test_batch_validation_is_as_strict_as_the_tools(tmp_path, monkeypatch):
    # pydantic strict rejects 2.0 for an integer; accepting it here would let the first
    # call move before the second fails inside its tool.
    schema = {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}
    executed, outcome = await run_fake(
        tmp_path, monkeypatch,
        [[call("1", "observe_panorama"), call("2", "execute_subtask", '{"task_id": 2.0}')],
         [call("3", "terminate_episode")]],
        names=["observe_panorama", "execute_subtask", "terminate_episode"],
        schemas={"execute_subtask": schema},
    )
    assert executed == ["terminate_episode"]
    assert outcome.error is None


async def test_only_the_current_call_can_end_the_episode(tmp_path, monkeypatch):
    stale = {"steps_taken_total": 3, "episode_over": True, "end_reason": "stop_called"}
    executed, outcome = await run_fake(
        tmp_path, monkeypatch,
        [[call("1", "observe_panorama")], [call("2", "terminate_episode")]],
        results={"observe_panorama": {"note": "no status"}},
        sink_state=stale,
    )
    assert executed == ["observe_panorama", "terminate_episode"]
    assert outcome.turns == 2
