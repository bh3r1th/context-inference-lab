"""GPU-free diagnostic tests; these fixtures do not establish CUDA compatibility."""

import subprocess
from types import SimpleNamespace

import pytest

from deploy.gcp import gpu_probe


def fake_gpu(monkeypatch, *, cuda_error=None, count=1, answer=2, smi_exit=0):
    events = []
    monkeypatch.setattr(gpu_probe.importlib.metadata, "version", lambda name: {
        "torch": "2.13.0+cu129", "vllm": "0.29.0",
    }[name])

    def smi(command, **kwargs):
        assert command == ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"]
        assert kwargs["timeout"] == 30 and kwargs["capture_output"] is True
        events.append("nvidia-smi")
        return subprocess.CompletedProcess(command, smi_exit, "NVIDIA L4, fixture-driver, 23034 MiB\n",
                                           "driver query failed" if smi_exit else "")

    def init():
        events.append("cuda.init")
        if cuda_error:
            raise RuntimeError(cuda_error)

    class Tensor:
        def __add__(self, value):
            assert value == 1
            events.append("cuda.kernel")
            return self

        def item(self):
            return answer

    def ones(size, *, device):
        assert size == 1 and device == "cuda:0"
        return Tensor()

    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.9"), ones=ones,
        cuda=SimpleNamespace(init=init, device_count=lambda: count,
                             get_device_properties=lambda _: SimpleNamespace(
                                 name="NVIDIA L4", total_memory=24 * 1024**3),
                             synchronize=lambda: events.append("cuda.synchronize")))

    def load(name):
        events.append(name)
        if name == "vllm.env_override":
            return SimpleNamespace()
        assert name == "torch"
        return torch

    monkeypatch.setattr(gpu_probe.subprocess, "run", smi)
    monkeypatch.setattr(gpu_probe.importlib, "import_module", load)
    return events


def test_probe_uses_server_compatibility_order_and_executes_gpu_operation(monkeypatch, capsys):
    events = fake_gpu(monkeypatch)
    assert gpu_probe.main() == 0
    assert events == ["nvidia-smi", "vllm.env_override", "torch", "cuda.init", "cuda.kernel",
                      "cuda.synchronize"]
    output = capsys.readouterr().out
    assert "fixture-driver" in output and '"torch_cuda_build": "12.9"' in output
    assert '"status": "passed"' in output


def test_cuda_803_fails_with_driver_diagnostics_and_no_kernel(monkeypatch, capsys):
    events = fake_gpu(monkeypatch, cuda_error="Error 803: unsupported driver / CUDA combination")
    assert gpu_probe.main() == 1
    captured = capsys.readouterr()
    assert "fixture-driver" in captured.out and "Error 803" in captured.err
    assert "cuda.kernel" not in events and '"status": "passed"' not in captured.out


@pytest.mark.parametrize("count,answer", [(0, 2), (1, 3)])
def test_missing_device_or_invalid_kernel_result_is_not_accepted(monkeypatch, count, answer):
    fake_gpu(monkeypatch, count=count, answer=answer)
    assert gpu_probe.main() == 1


def test_failed_driver_query_does_not_import_cuda_libraries(monkeypatch, capsys):
    events = fake_gpu(monkeypatch, smi_exit=1)
    assert gpu_probe.main() == 1
    assert events == ["nvidia-smi"]
    assert "driver query failed" in capsys.readouterr().err


def test_missing_nvidia_smi_is_a_clear_failure(monkeypatch, capsys):
    fake_gpu(monkeypatch)

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi is not installed")

    monkeypatch.setattr(gpu_probe.subprocess, "run", missing)
    assert gpu_probe.main() == 1
    assert "nvidia-smi is not installed" in capsys.readouterr().err


def compatibility_changes_library_path(monkeypatch):
    events = fake_gpu(monkeypatch)
    loader = gpu_probe.importlib.import_module

    def load(name):
        value = loader(name)
        if name == "vllm.env_override":
            monkeypatch.setenv("LD_LIBRARY_PATH", "/fixture/cuda-compat:/fixture/native")
        return value

    monkeypatch.setattr(gpu_probe.importlib, "import_module", load)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fixture/native")
    return events


def test_changed_compatibility_path_reexecutes_before_cuda_initialization(monkeypatch):
    events = compatibility_changes_library_path(monkeypatch)
    monkeypatch.delenv(gpu_probe.REEXEC_MARKER, raising=False)
    monkeypatch.setenv("CONTEXTBENCH_PROBE_TEST", "retained")
    calls = []

    class ExecRequested(BaseException):
        pass

    def execute(path, arguments, environment):
        calls.append((path, arguments, environment))
        raise ExecRequested

    monkeypatch.setattr(gpu_probe.os, "execve", execute)
    with pytest.raises(ExecRequested):
        gpu_probe.check_gpu()
    assert events == ["nvidia-smi", "vllm.env_override"]
    path, arguments, environment = calls[0]
    assert path == gpu_probe.sys.executable
    assert arguments == [path, "-m", "deploy.gcp.gpu_probe"]
    assert environment["LD_LIBRARY_PATH"] == "/fixture/cuda-compat:/fixture/native"
    assert environment[gpu_probe.REEXEC_MARKER] == "1"
    assert environment["CONTEXTBENCH_PROBE_TEST"] == "retained"


def test_reexecuted_probe_does_not_loop_when_compatibility_changes_path_again(monkeypatch):
    events = compatibility_changes_library_path(monkeypatch)
    monkeypatch.setenv(gpu_probe.REEXEC_MARKER, "1")

    def unexpected_exec(*_args):
        raise AssertionError("The probe must re-execute at most once")

    monkeypatch.setattr(gpu_probe.os, "execve", unexpected_exec)
    assert gpu_probe.main() == 0
    assert "cuda.init" in events and "cuda.synchronize" in events
