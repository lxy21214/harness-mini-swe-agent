from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from minisweagent.exceptions import Submitted
from minisweagent.models.streaming_litellm_model import (
    HTTPTimeoutConfig,
    StreamingLitellmModel,
    model_architecture,
)


def _response(tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _model(tmp_path: Path, **kwargs):
    model_name = kwargs.pop("model_name", "@provider/model")
    return StreamingLitellmModel(
        model_name=model_name,
        base_url="https://model.example/v1",
        api_key="test-key",
        api_calls_log=str(tmp_path / "api_calls.log"),
        **kwargs,
    )


def _priced_response(
    prompt_tokens,
    cached_tokens,
    completion_tokens,
    cache_write_tokens=0,
    input_tokens=None,
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    if cache_write_tokens:
        usage.cache_creation_input_tokens = cache_write_tokens
    if input_tokens is not None:
        usage.input_tokens = input_tokens
    return SimpleNamespace(usage=usage)


@pytest.mark.parametrize(
    ("model_name", "architecture"),
    [
        ("openai/@relay/DeepSeek-V4-Pro", "deepseek-v4-pro"),
        ("@relay/DeepSeek-V4-Flash", "deepseek-v4-flash"),
        ("openai/@relay/Claude-Sonnet-4.6", "claude-sonnet-4.6"),
        ("openai/@relay/Claude-Sonnet-4-6", "claude-sonnet-4.6"),
        ("openai/gpt-5.6-sol", "gpt-5.6-sol"),
        ("openai/@relay/GLM5.2", "glm-5.2"),
        ("openai/@relay/GLM-5.2", "glm-5.2"),
        ("openai/k3", "kimi-k3"),
        ("@relay/K3-256K", "kimi-k3"),
        ("openai/@relay/Kimi-K3", "kimi-k3"),
        ("openai/@relay/Kimi-K3-256K", "kimi-k3"),
        ("openai/@relay/MiniMax/MiniMax-M3", "minimax-m3"),
        ("vendor/Claude-Sonnet-4.5-20250929", "claude-sonnet-4.5"),
        ("vendor/GPT-5.5-pro-thinking", "gpt-5.5-pro"),
        ("vendor/GLM.5-2-preview", "glm-5.2"),
    ],
)
def test_model_suffix_maps_to_official_architecture(model_name, architecture):
    assert model_architecture(model_name) == architecture


@pytest.mark.parametrize(
    ("model_name", "expected_cost"),
    [
        ("openai/@relay/deepseek-v4-pro", 0.2662),
        ("openai/@relay/deepseek-v4-flash", 0.0887),
        ("openai/@relay/glm5.2", 0.606),
        ("openai/k3", 1.83),
        ("openai/@relay/k3-256k", 1.83),
        ("openai/@relay/kimi-k3", 1.83),
        ("openai/@relay/kimi-k3-256k", 1.83),
        ("openai/@relay/minimax-m3", 0.156),
    ],
)
def test_cost_uses_official_architecture_prices(tmp_path, model_name, expected_cost):
    model = _model(tmp_path, model_name=model_name)
    response = _priced_response(
        prompt_tokens=200_000,
        cached_tokens=100_000,
        completion_tokens=100_000,
    )

    assert model._calculate_cost(response)["cost"] == pytest.approx(expected_cost)


def test_minimax_m3_uses_long_context_price_above_512k(tmp_path):
    model = _model(tmp_path, model_name="openai/@relay/minimax-m3")
    response = _priced_response(
        prompt_tokens=600_000,
        cached_tokens=100_000,
        completion_tokens=100_000,
    )

    assert model._calculate_cost(response)["cost"] == pytest.approx(0.552)


def test_new_rate_rows_include_cache_write_pricing(tmp_path):
    model = _model(tmp_path, model_name="openai/@relay/claude-sonnet-5")
    response = _priced_response(
        prompt_tokens=100_000,
        cached_tokens=20_000,
        cache_write_tokens=10_000,
        completion_tokens=10_000,
    )

    # (70K * 2.0 + 20K * .2 + 10K * 2.5 + 10K * 10.0) / 1M
    assert model._calculate_cost(response)["cost"] == pytest.approx(0.269)


def test_anthropic_style_separate_input_and_cache_usage_is_supported(tmp_path):
    model = _model(tmp_path, model_name="openai/@relay/claude-sonnet-5")
    response = _priced_response(
        prompt_tokens=0,
        cached_tokens=20_000,
        cache_write_tokens=10_000,
        input_tokens=70_000,
        completion_tokens=10_000,
    )

    assert model._calculate_cost(response)["cost"] == pytest.approx(0.269)


def test_context_tier_changes_at_the_declared_boundary(tmp_path):
    model = _model(tmp_path, model_name="openai/gpt-5.6-sol")
    short_context = _priced_response(271_999, 0, 0)
    long_context = _priced_response(272_000, 0, 0)

    assert model._calculate_cost(short_context)["cost"] == pytest.approx(1.359995)
    assert model._calculate_cost(long_context)["cost"] == pytest.approx(2.72)


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
    assert "| APICalls | INFO" in log
    assert "test-key" not in log


def test_streaming_model_can_mirror_api_calls_to_console(tmp_path, capsys):
    model = _model(tmp_path, api_calls_console=True)

    class FakeResponse:
        def __str__(self):
            return "ModelResponse(id=console-response)"

    with (
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.completion",
            return_value=iter([object()]),
        ),
        patch(
            "minisweagent.models.streaming_litellm_model.litellm.stream_chunk_builder",
            return_value=FakeResponse(),
        ),
    ):
        model._query([{"role": "user", "content": "hello"}])

    console = capsys.readouterr().err
    assert "| APICalls | INFO" in console
    assert "ModelResponse(id=console-response)" in console


def test_five_consecutive_no_bash_responses_finish(tmp_path):
    model = _model(tmp_path)
    for _ in range(4):
        assert model._parse_actions(_response()) == []

    with pytest.raises(Submitted) as raised:
        model._parse_actions(_response())

    message = raised.value.messages[0]
    assert message["extra"]["exit_status"] == "NoBashCompletion"
    assert "5 consecutive" in message["content"]
    log = (tmp_path / "api_calls.log").read_text(encoding="utf-8")
    assert "Completed mini-swe-agent after 5 consecutive" in log


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
