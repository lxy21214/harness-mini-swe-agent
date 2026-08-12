import os

import pytest

from minisweagent.environments.local import LocalEnvironment


@pytest.mark.skipif(os.name == "nt", reason="shell executable paths are POSIX-specific")
def test_local_environment_uses_shell_from_environment():
    """The command interpreter is selected only through the supplied SHELL value."""
    env = LocalEnvironment(env={"SHELL": "/bin/bash"})

    result = env.execute({"command": "if [[ 1 -eq 1 ]]; then echo bash; fi"})

    assert result["returncode"] == 0
    assert result["output"].strip() == "bash"
