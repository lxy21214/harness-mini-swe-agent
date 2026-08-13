import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.environments.local import LocalEnvironment, _run


@pytest.mark.skipif(os.name == "nt", reason="shell executable paths are POSIX-specific")
def test_local_environment_uses_shell_from_environment():
    """The command interpreter is selected only through the supplied SHELL value."""
    env = LocalEnvironment(env={"SHELL": "/bin/bash"})

    result = env.execute({"command": "if [[ 1 -eq 1 ]]; then echo bash; fi"})

    assert result["returncode"] == 0
    assert result["output"].strip() == "bash"


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-specific")
def test_timeout_does_not_wait_for_stdout_after_kill():
    process = MagicMock(pid=1234)
    process.communicate.side_effect = subprocess.TimeoutExpired(
        "command", 1, output="partial output"
    )

    with (
        patch("minisweagent.environments.local.subprocess.Popen", return_value=process),
        patch("minisweagent.environments.local.os.killpg") as killpg,
        pytest.raises(subprocess.TimeoutExpired) as raised,
    ):
        _run("command", "/tmp", {}, 1)

    process.communicate.assert_called_once_with(timeout=1)
    killpg.assert_called_once_with(1234, 9)
    assert raised.value.output == "partial output"
