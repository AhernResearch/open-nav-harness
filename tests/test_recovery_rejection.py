import json
import pytest
from nav_harness.core.recovery import resource_state


@pytest.mark.parametrize('wire', [True, False])
def test_only_explicit_no_motion_rejection_releases_pending_call(wire):
    dispatch = {'kind': 'tool_dispatch' if wire else 'tool_use', 'name': 'execute_subtask',
                'call_id' if wire else 'id': 'one'}
    for data, expected in [({'error': 'precondition_failed', 'state_changed': False}, 'idle'),
                           ({'error': 'precondition_failed'}, 'unknown'),
                           ({'error': 'precondition_failed', 'state_changed': True}, 'unknown'),
                           ({'error': 'timeout', 'state_changed': False}, 'unknown')]:
        result = {'kind': 'tool_result', 'is_error': True, 'call_id' if wire else 'tool_use_id': 'one'}
        result.update({'feedback': {'content': json.dumps(data)}} if wire else {'texts': [json.dumps(data)]})
        assert resource_state([dispatch, result], wire=wire) == expected
        assert resource_state([dispatch, result], wire=wire,
                              worker_info={'protocol_version': 1, 'session_id': 'active'}) == 'unknown'
    assert resource_state([dispatch], wire=wire) == 'unknown'
