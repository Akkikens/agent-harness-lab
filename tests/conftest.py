import os
import subprocess

import pytest

from harness.sandbox import DockerSandbox


@pytest.fixture(scope="session")
def docker_box():
    if not DockerSandbox.daemon_up():
        pytest.skip("Docker daemon not running")
    box = DockerSandbox()
    try:
        box.ensure_image()
    except subprocess.CalledProcessError:
        pytest.skip("cannot pull sandbox image (offline?)")
    return box


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d.resolve()
