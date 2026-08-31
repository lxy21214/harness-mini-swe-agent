"""A configuration-driven streaming LiteLLM model.

This model is intentionally separate from :class:`LitellmModel`.  It keeps
stream assembly and API-call logging local to the model implementation while
using the normal mini-SWE-agent model interface.
"""

from __future__ import annotations

import datetime
import logging
import logging.handlers
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import litellm
from pydantic import BaseModel, Field, SecretStr

from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX

# Model-name regex → openrouter provider pinned via provider.only.
_PROVIDER_ONLY_ROUTING: list[tuple[str, str]] = [
    (r"\bgpt\b", "OpenAI"),
    (r"gemini", "google-vertex"),
]


@dataclass(frozen=True)
class OfficialTokenRates:
    """Official USD list prices per one million tokens."""
    input: float
    cached_input: float
    output: float
    cache_write: float = 0.0


@dataclass(frozen=True)
class OfficialRateTier:
    """A price tier selected by the request's input-context token count."""

    max_context_tokens: int
    rates: OfficialTokenRates


def _rate(
    input: float,
    output: float,
    cache_input: float | None = None,
    cache_write: float = 0.0,
) -> OfficialTokenRates:
    """Build a rate row using the hand-maintained table's field names.

    Prices are USD per one million tokens.  Models without a ``cache_input``
    price do not advertise a cache discount, so cached input falls back to the
    regular input price.  ``cache_write`` is zero when the provider does not
    charge a separate cache-write rate.
    """
    return OfficialTokenRates(
        input=input,
        cached_input=input if cache_input is None else cache_input,
        output=output,
        cache_write=cache_write,
    )


def _tier(
    max_context_tokens: int,
    input: float,
    output: float,
    cache_input: float | None = None,
    cache_write: float = 0.0,
) -> OfficialRateTier:
    return OfficialRateTier(
        max_context_tokens=max_context_tokens,
        rates=_rate(input, output, cache_input, cache_write),
    )


# Public standard-tier prices used for paper reporting, not necessarily the
# prices charged by a relay.  Each row is USD per one million tokens.  The
# upper bound follows the table's ``[lower-upper)`` convention.
OFFICIAL_MODEL_RATES: dict[str, tuple[OfficialRateTier, ...]] = {
    "claude-fable-5": (_tier(1_000_000, 10.0, 50.0, 1.0, 12.5),),
    "claude-opus-5": (_tier(1_000_000, 5.0, 25.0, 0.5, 6.25),),
    "claude-sonnet-5": (_tier(1_000_000, 2.0, 10.0, 0.2, 2.5),),
    "claude-opus-4.8": (_tier(1_000_000, 5.0, 25.0, 0.5, 6.25),),
    "claude-opus-4.7": (_tier(1_000_000, 5.0, 25.0, 0.5, 6.25),),
    "claude-opus-4.6": (_tier(1_000_000, 5.0, 25.0, 0.5, 6.25),),
    "claude-sonnet-4.6": (_tier(1_000_000, 3.0, 15.0, 0.3, 3.75),),
    "claude-opus-4.5": (_tier(1_000_000, 5.0, 25.0, 0.5, 6.25),),
    "claude-sonnet-4.5": (
        _tier(200_000, 3.0, 15.0, 0.3, 3.75),
        _tier(1_000_000, 6.0, 22.5, 0.6, 7.5),
    ),
    "claude-haiku-4.5": (_tier(1_000_000, 1.0, 5.0, 0.1, 1.25),),
    "gpt-5.6-sol": (
        _tier(272_000, 5.0, 30.0, 0.5, 6.25),
        _tier(1_048_576, 10.0, 45.0, 1.0, 12.5),
    ),
    "gpt-5.6-terra": (
        _tier(272_000, 2.0, 12.0, 0.2, 2.5),
        _tier(1_048_576, 4.0, 18.0, 0.4, 5.0),
    ),
    "gpt-5.6-luna": (
        _tier(272_000, 0.2, 1.2, 0.02, 0.25),
        _tier(1_048_576, 0.4, 1.8, 0.04, 0.5),
    ),
    "gpt-5.5-pro": (
        _tier(272_000, 30.0, 180.0),
        _tier(1_048_576, 60.0, 270.0),
    ),
    "gpt-5.5": (
        _tier(272_000, 5.0, 30.0, 0.5),
        _tier(1_048_576, 10.0, 45.0, 1.0),
    ),
    "gpt-5.4-pro": (
        _tier(272_000, 30.0, 180.0),
        _tier(1_048_576, 60.0, 270.0),
    ),
    "gpt-5.4": (
        _tier(272_000, 2.5, 15.0, 0.25),
        _tier(1_048_576, 5.0, 22.5, 0.5),
    ),
    "gpt-5.4-mini": (_tier(400_000, 0.75, 4.5, 0.075),),
    "gpt-5.4-nano": (_tier(400_000, 0.2, 1.25, 0.02),),
    "gemini-3.7-flash": (_tier(1_048_576, 1.5, 7.5, 0.15, 0.08333),),
    "gemini-3.6-flash": (_tier(1_048_576, 0.75, 3.75, 0.075, 0.04167),),
    "gemini-3.5-flash": (_tier(1_048_576, 1.5, 9.0, 0.15, 0.08333),),
    "gemini-3.5-flash-lite": (_tier(1_048_576, 0.3, 2.5, 0.03, 0.08333),),
    "gemini-3.1-flash-lite": (_tier(1_048_576, 0.25, 1.5, 0.025, 0.08333),),
    "gemini-3.1-pro-preview": (
        _tier(200_000, 2.0, 12.0, 0.2, 0.375),
        _tier(1_048_576, 4.0, 18.0, 0.4, 0.375),
    ),
    "grok-4.6": (
        _tier(200_000, 2.0, 6.0, 0.5),
        _tier(500_000, 4.0, 12.0, 1.0),
    ),
    "grok-4.5": (
        _tier(200_000, 2.0, 6.0, 0.3),
        _tier(500_000, 4.0, 12.0, 0.6),
    ),
    "glm-5.3": (_tier(1_048_576, 1.40, 4.40, 0.26),),
    "glm-5.2": (_tier(1_048_576, 1.40, 4.40, 0.26),),
    "kimi-k3": (_tier(1_048_576, 3.00, 15.00, 0.30),),
    "kimi-k2.7-code": (_tier(262_144, 0.95, 4.0, 0.19),),
    "deepseek-v4-flash": (_tier(1_000_000, 0.22, 0.66, 0.007),),
    "deepseek-v4-pro": (_tier(1_000_000, 0.66, 1.98, 0.022),),
    "qwen3.8-max": (_tier(1_000_000, 1.65, 4.951, 0.206),),
    "qwen3.8-2.4t-a95b": (_tier(1_000_000, 1.65, 4.951, 0.206),),
    "qwen3.8-27b": (_tier(1_000_000, 0.424, 1.696, 0.085),),
    "qwen3.7-max": (_tier(1_000_000, 1.65, 4.951, 0.33),),
    "qwen3.7-plus": (
        _tier(256_000, 0.276, 1.101, 0.056),
        _tier(1_000_000, 0.826, 3.301, 0.166),
    ),
    "qwen3.7-flash": (
        _tier(32_000, 0.028, 0.11, 0.006),
        _tier(256_000, 0.083, 0.33, 0.017),
        _tier(1_000_000, 0.165, 0.66, 0.033),
    ),
    "qwen3.6-max": (
        _tier(128_000, 1.027, 6.162),
        _tier(256_000, 1.58, 9.48),
    ),
    "qwen3.6-plus": (
        _tier(256_000, 0.1, 0.4),
        _tier(1_000_000, 0.2, 0.8),
    ),
    "qwen3.6-flash": (
        _tier(256_000, 0.165, 0.99),
        _tier(1_000_000, 0.66, 3.961),
    ),
    "qwen3.6-27b": (_tier(256_000, 0.412564, 2.475384),),
    "qwen3.6-35b-a3b": (_tier(256_000, 0.248, 1.485),),
    "qwen3.5-plus": (_tier(256_000, 0.115, 0.688),),
    "qwen3.5-flash": (_tier(256_000, 0.029, 0.287),),
    "qwen3.5-397b-a17b": (
        _tier(128_000, 0.172, 1.032),
        _tier(256_000, 0.43, 2.58),
    ),
    "qwen3.5-122b-a10b": (
        _tier(128_000, 0.115, 0.917),
        _tier(256_000, 0.287, 2.294),
    ),
    "qwen3.5-27b": (
        _tier(128_000, 0.086, 0.688),
        _tier(256_000, 0.258, 2.064),
    ),
    "qwen3.5-35b-a3b": (
        _tier(128_000, 0.057, 0.459),
        _tier(256_000, 0.229, 1.835),
    ),
    "minimax-m3": (
        _tier(512_000, 0.3, 1.2, 0.06),
        _tier(1_000_000, 0.6, 2.4, 0.12),
    ),
    "mimo-v2.5-pro": (_tier(1_048_576, 0.435, 0.87, 0.0036),),
    "mimo-v2.5": (_tier(1_048_576, 0.14, 0.28, 0.0028),),
    "seed-2-1-turbo": (_tier(262_144, 0.5, 2.5, 0.1),),
    "seed-2.0-pro": (
        _tier(131_072, 0.5, 3.0, 0.1),
        _tier(262_144, 1.0, 6.0, 0.2),
    ),
    "seed-2.0-code": (
        _tier(131_072, 0.5, 3.0, 0.1),
        _tier(262_144, 1.0, 6.0, 0.2),
    ),
    "seed-2.0-lite": (
        _tier(131_072, 0.25, 2.0, 0.05),
        _tier(262_144, 0.5, 4.0, 0.1),
    ),
    "seed-2.0-mini": (
        _tier(131_072, 0.1, 0.4, 0.02),
        _tier(262_144, 0.2, 0.8, 0.04),
    ),
    "hy3": (_tier(262_144, 0.132, 0.528, 0.033),),
}


# Relay/provider prefixes are intentionally allowed.  Model identifiers are
# matched by normalized substring rather than exact equality because relays
# commonly append provider, date, or capability suffixes.  Normalization is
# deliberately shared by aliases and incoming model names: separators are
# folded to underscores after applying the requested upper/lower conversion.
MODEL_ARCHITECTURES: dict[str, str] = {
    "glm5.2": "glm-5.2",
    "k3": "kimi-k3",
    "k3-256k": "kimi-k3",
    "kimi-k3-256k": "kimi-k3",
}
for _architecture in OFFICIAL_MODEL_RATES:
    MODEL_ARCHITECTURES.setdefault(_architecture, _architecture)
    MODEL_ARCHITECTURES.setdefault(_architecture.replace(".", "-"), _architecture)


def _normalize_model_name(model_name: str) -> str:
    return model_name.upper().lower().replace(".", "_").replace("-", "_")


_NORMALIZED_MODEL_ARCHITECTURES: dict[str, str] = {
    _normalize_model_name(alias): architecture
    for alias, architecture in MODEL_ARCHITECTURES.items()
}
_MODEL_ARCHITECTURE_MATCHES = tuple(
    sorted(
        _NORMALIZED_MODEL_ARCHITECTURES.items(),
        # Prefer the most specific (longest) substring, e.g. ``gpt_5_5_pro``
        # before the shorter ``gpt_5_5`` alias.
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def model_architecture(model_name: str) -> str | None:
    normalized_name = _normalize_model_name(model_name)
    for normalized_alias, architecture in _MODEL_ARCHITECTURE_MATCHES:
        if normalized_alias in normalized_name:
            return architecture
    return None


def _optional_token_count(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


def _token_count(source: Any, *names: str) -> float:
    """Return the largest usable count across provider-specific field names."""
    values: list[float] = []
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if (number := _optional_token_count(value)) is not None:
            values.append(number)
    return max(values, default=0.0)


def _select_rate_tier(
    architecture: str,
    context_tokens: float,
) -> OfficialTokenRates:
    tiers = OFFICIAL_MODEL_RATES[architecture]
    for tier in tiers:
        if context_tokens < tier.max_context_tokens:
            return tier.rates
    # The supplied table stops at each model's documented context limit.  If a
    # provider reports a request exactly at/above that limit, keep using the
    # final tier rather than dropping cost accounting for the response.
    return tiers[-1].rates


class HTTPTimeoutConfig(BaseModel):
    """The four timeout values accepted by ``httpx.Timeout``."""

    connect: float = 30.0
    read: float = 90.0
    write: float = 30.0
    pool: float = 30.0

    def build(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect,
            read=self.read,
            write=self.write,
            pool=self.pool,
        )


class NoBashConfig(BaseModel):
    """Limits for model responses that contain no bash tool call."""

    consecutive: int = 5
    total: int = 25


class StreamingLitellmModelConfig(LitellmModelConfig):
    """Configuration read from the model section of a YAML config file."""

    # Keep these settings explicit and configuration-driven.  API parameters
    # come from the config file; MINISWEA_TEMPERATURE / MINISWEA_MAX_TOKENS
    # environment variables may override temperature / max_tokens (the env
    # default is used only when the config omits the field).
    base_url: str
    api_key: SecretStr
    temperature: float = Field(
        default_factory=lambda: float(os.environ.get("MINISWEA_TEMPERATURE", "0.3"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("MINISWEA_MAX_TOKENS", "131072"))
    )
    timeout: HTTPTimeoutConfig | float = Field(default_factory=HTTPTimeoutConfig)
    no_bash: NoBashConfig = Field(default_factory=NoBashConfig)
    api_calls_log: str | None = ".logs/api_calls.log"
    api_calls_console: bool = False
    allow_partial_stream: bool = True
    multimodal_regex: str = DEFAULT_MULTIMODAL_REGEX

    # Do not inherit environment-driven defaults for the optional registry and
    # cost mode when this model is selected from a config file.
    litellm_model_registry: Path | str | None = None
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"


class StreamingLitellmModel(LitellmModel):
    """Stream LiteLLM responses, record them, and stop repeated no-bash turns."""

    def __init__(self, **kwargs):
        super().__init__(config_class=StreamingLitellmModelConfig, **kwargs)
        self._consecutive_no_bash_responses = 0
        self._total_no_bash_responses = 0
        self._api_logger = self._make_api_logger(
            self.config.api_calls_log,
            self.config.api_calls_console,
        )

    @staticmethod
    def _make_api_logger(log_path: str | None, console: bool) -> logging.Logger:
        if log_path:
            path = Path(log_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            logger_name = f"minisweagent.models.streaming_litellm.api_calls.{path}"
        else:
            path = None
            logger_name = "minisweagent.models.streaming_litellm.api_calls.disabled"

        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s | APICalls | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if path and not any(
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and Path(handler.baseFilename) == path
            for handler in logger.handlers
        ):
            handler = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if console and not any(
            getattr(handler, "_mswea_api_calls_console", False)
            for handler in logger.handlers
        ):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            handler._mswea_api_calls_console = True  # type: ignore[attr-defined]
            logger.addHandler(handler)

        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    def _timeout(self) -> httpx.Timeout:
        if isinstance(self.config.timeout, HTTPTimeoutConfig):
            return self.config.timeout.build()
        return httpx.Timeout(self.config.timeout)

    def _calculate_cost(self, response) -> dict[str, float]:
        architecture = model_architecture(self.config.model_name)
        usage = getattr(response, "usage", None)
        if architecture is None or usage is None:
            return super()._calculate_cost(response)

        prompt_tokens = max(float(getattr(usage, "prompt_tokens", 0) or 0), 0.0)
        completion_tokens = max(float(getattr(usage, "completion_tokens", 0) or 0), 0.0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = max(
            _token_count(prompt_details, "cached_tokens"),
            _token_count(
                usage,
                "prompt_cache_hit_tokens",
                "cache_read_input_tokens",
            ),
        )
        cache_write_tokens = max(
            _token_count(prompt_details, "cache_creation_input_tokens", "cache_write_tokens"),
            _token_count(
                usage,
                "cache_creation_input_tokens",
                "prompt_cache_creation_tokens",
                "cache_write_input_tokens",
                "cache_write_tokens",
            ),
        )
        has_separate_cache_usage = bool(
            _token_count(
                prompt_details,
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "cache_write_tokens",
            )
            or _token_count(
                usage,
                "cache_creation_input_tokens",
                "prompt_cache_creation_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
                "cache_write_tokens",
            )
        )

        # Anthropic-style responses report uncached ``input_tokens`` and cache
        # reads/writes as separate fields.  OpenAI-style responses report one
        # aggregate ``prompt_tokens`` count, so cache tokens are subtracted from
        # that aggregate instead.
        provider_input_tokens = _optional_token_count(
            getattr(usage, "input_tokens", None)
        )
        if provider_input_tokens is not None and has_separate_cache_usage:
            uncached_tokens = provider_input_tokens
            context_tokens = (
                uncached_tokens + cached_tokens + cache_write_tokens
            )
        else:
            cached_tokens = min(cached_tokens, prompt_tokens)
            cache_write_tokens = min(
                cache_write_tokens,
                max(prompt_tokens - cached_tokens, 0.0),
            )
            uncached_tokens = max(
                prompt_tokens - cached_tokens - cache_write_tokens,
                0.0,
            )
            context_tokens = prompt_tokens

        rates = _select_rate_tier(architecture, context_tokens)

        cost = (
            uncached_tokens * rates.input
            + cached_tokens * rates.cached_input
            + cache_write_tokens * rates.cache_write
            + completion_tokens * rates.output
        ) / 1_000_000
        return {"cost": cost}

    def _query(self, messages: list[dict[str, str]], **kwargs):
        start_time = datetime.datetime.now(datetime.UTC)
        chunks = []
        request_kwargs: dict[str, Any] = dict(self.config.model_kwargs)
        request_kwargs.update(kwargs)
        request_kwargs.update(
            {
                "model": self.config.model_name,
                "messages": messages,
                "tools": [BASH_TOOL],
                "base_url": self.config.base_url,
                "api_key": self.config.api_key.get_secret_value(),
                "stream": True,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self._timeout(),
            }
        )

        # Pin model families to their provider via openrouter provider
        # selection (https://openrouter.ai/docs/guides/routing/provider-selection).
        # The detection is based on the model name only: the devbox does not
        # expose the upstream base URL, so we cannot tell whether the request
        # goes through openrouter and must not try to.
        for pattern, provider_name in _PROVIDER_ONLY_ROUTING:
            if re.search(pattern, self.config.model_name, re.IGNORECASE):
                request_kwargs.setdefault("provider", {"only": [provider_name]})
                break

        try:
            for chunk in litellm.completion(**request_kwargs):
                chunks.append(chunk)
        except Exception:
            if not (self.config.allow_partial_stream and chunks):
                self._api_logger.exception("Streaming model call failed")
                raise
            self._api_logger.warning("Streaming model call ended after a partial response", exc_info=True)

        response = litellm.stream_chunk_builder(
            chunks,
            messages,
            start_time=start_time,
            end_time=datetime.datetime.now(datetime.UTC),
        )
        self._api_logger.info("%s", response)
        return response

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            self._consecutive_no_bash_responses += 1
            self._total_no_bash_responses += 1
            consecutive = self._consecutive_no_bash_responses
            total = self._total_no_bash_responses
            self._api_logger.warning(
                "Model response contained no bash call "
                "(consecutive %d/%d, total %d/%d); finish_reason=%s",
                consecutive,
                self.config.no_bash.consecutive,
                total,
                self.config.no_bash.total,
                response.choices[0].finish_reason,
            )
            if (
                self.config.no_bash.consecutive > 0
                and consecutive >= self.config.no_bash.consecutive
            ):
                raise self._no_bash_completion(
                    f"{consecutive} consecutive model responses without a bash call"
                )
            if self.config.no_bash.total > 0 and total >= self.config.no_bash.total:
                raise self._no_bash_completion(
                    f"{total} total model responses without a bash call"
                )
            return []

        self._consecutive_no_bash_responses = 0
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def _no_bash_completion(self, reason: str) -> Submitted:
        message = f"Completed mini-swe-agent after {reason}"
        self._api_logger.info("%s", message)
        return Submitted(
            {
                "role": "exit",
                "content": message,
                "extra": {
                    "exit_status": "NoBashCompletion",
                    "submission": "",
                },
            }
        )


# Keep a short name for config files that already use the previous class name.
StreamLitellmModel = StreamingLitellmModel
