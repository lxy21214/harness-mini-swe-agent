"""A configuration-driven streaming LiteLLM model.

This model is intentionally separate from :class:`LitellmModel`.  It keeps
stream assembly and API-call logging local to the model implementation while
using the normal mini-SWE-agent model interface.
"""

from __future__ import annotations

import datetime
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Literal

import httpx
import litellm
from pydantic import BaseModel, Field, SecretStr

from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX


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

    # Keep these settings explicit and configuration-driven.  In particular,
    # this model does not inspect environment variables for API parameters.
    base_url: str
    api_key: SecretStr
    temperature: float = 0.3
    max_tokens: int = 131072
    timeout: HTTPTimeoutConfig | float = Field(default_factory=HTTPTimeoutConfig)
    no_bash: NoBashConfig = Field(default_factory=NoBashConfig)
    api_calls_log: str | None = ".logs/api_calls.log"
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
        self._api_logger = self._make_api_logger(self.config.api_calls_log)

    @staticmethod
    def _make_api_logger(log_path: str | None) -> logging.Logger:
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
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        return logger

    def _timeout(self) -> httpx.Timeout:
        if isinstance(self.config.timeout, HTTPTimeoutConfig):
            return self.config.timeout.build()
        return httpx.Timeout(self.config.timeout)

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

    @staticmethod
    def _no_bash_completion(reason: str) -> Submitted:
        message = f"Completed mini-swe-agent after {reason}"
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
