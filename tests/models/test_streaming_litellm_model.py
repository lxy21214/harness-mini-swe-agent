from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from minisweagent.exceptions import Submitted
from minisweagent.models.streaming_litellm_model import (
    HTTPTimeoutConfig,
    StreamingLitellmModel,
)


def _response(tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _model(tmp_path: Path, **kwargs):
    return StreamingLitellmModel(
        model_name="@provider/model",
        base_url="https://model.example/v1",
        api_key="test-key",
        api_calls_log=str(tmp_path / "api_calls.log"),
        **kwargs,
    )


def test_model_parameters_are_taken_verbatim_from_config(tmp_path):
    model = _model(
        tmp_path,
        temperature=0.7,
        max_tokens=4096,
        timeout=HTTPTimeoutConfig(connect=1, read=2, write=3, pool=4),
    )
    chunks = [object()]
    assembled = MagicMock()

    with (
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.completion",
            return_value=iter(chunks),
        ) as completion,
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.stream_chunk_builder",
            return_value=assembled,
        ),
    ):
        assert model._query([{"role": "user", "content": "hello"}]) is assembled

    request = completion.call_args.kwargs
    assert request["model"] == "@provider/model"
    assert not request["model"].startswith("openai/")
    assert request["base_url"] == "https://model.example/v1"
    assert request["api_key"] == "test-key"
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 4096
    assert request["stream"] is True
    assert isinstance(request["timeout"], httpx.Timeout)
    assert request["timeout"].connect == 1
    assert request["timeout"].read == 2
    assert request["timeout"].write == 3
    assert request["timeout"].pool == 4


def test_streaming_model_records_api_calls_only_in_its_configured_log(tmp_path):
    model = _model(tmp_path)

    class FakeResponse:
        def __str__(self):
            return "ModelResponse(id=test-response)"

    response = FakeResponse()

    with (
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.completion",
            return_value=iter([object()]),
        ),
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.stream_chunk_builder",
            return_value=response,
        ),
    ):
        model._query([{"role": "user", "content": "hello"}])

    log = (tmp_path / "api_calls.log").read_text(encoding="utf-8")
    assert "ModelResponse" in log
    assert "test-key" not in log


def test_five_consecutive_no_bash_responses_finish(tmp_path):
    model = _model(tmp_path)
    for _ in range(4):
        assert model._parse_actions(_response()) == []

    with pytest.raises(Submitted) as raised:
        model._parse_actions(_response())

    message = raised.value.messages[0]
    assert message["extra"]["exit_status"] == "NoBashCompletion"
    assert "5 consecutive" in message["content"]


def test_bash_resets_consecutive_count_but_not_total(tmp_path):
    tool_call = object()
    model = _model(tmp_path)
    for _ in range(3):
        assert model._parse_actions(_response()) == []
    with patch(
        "minisweagent.models.streaming_litellm_model.parse_toolcall_actions",
        return_value=[{"command": "echo ok"}],
    ):
        assert model._parse_actions(_response([tool_call])) == [{"command": "echo ok"}]
    assert model._consecutive_no_bash_responses == 0
    assert model._total_no_bash_responses == 3


def test_total_no_bash_limit_is_configurable(tmp_path):
    model = _model(
        tmp_path,
        no_bash={"consecutive": 0, "total": 2},
    )
    model._parse_actions(_response())
    with pytest.raises(Submitted):
        model._parse_actions(_response())
