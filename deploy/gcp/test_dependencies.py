"""Reject unrelated metadata failures while retaining vLLM's documented NCCL stack."""

import subprocess

import pytest

from deploy.gcp import check_dependencies as check

# Preserve the actual Cloud Build regression independently of the checker constant.
BUILD_DIAGNOSTIC = (
    'torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; '
    'platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.\n'
)
VERSIONS = {"torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7", "vllm": "0.29.0"}
PIN = "nvidia-nccl-cu13==2.30.7\n"


def test_normal_healthy_environment_needs_no_exception():
    assert check.validate_pip_check(0, "No broken requirements found.\n", "") is False


def test_actual_build_conflict_requires_vendor_proof():
    assert check.validate_pip_check(1, BUILD_DIAGNOSTIC, "", VERSIONS, PIN) is True


@pytest.mark.parametrize("diagnostic", [
    BUILD_DIAGNOSTIC + "contextbench 0.1.0 requires httpx, which is not installed.\n",
    "contextbench 0.1.0 requires httpx, which is not installed.\n",
    BUILD_DIAGNOSTIC.replace("but you have nvidia-nccl-cu13 2.30.7",
                             "but you have nvidia-nccl-cu13 2.31.0"),
    BUILD_DIAGNOSTIC.replace("nvidia-nccl-cu13==2.29.7;", "nvidia-nccl-cu13==2.28.0;"),
    "",
])
def test_other_or_additional_conflicts_are_never_hidden(diagnostic):
    with pytest.raises(check.DependencyValidationError, match="unrecognized"):
        check.validate_pip_check(1, diagnostic, "", VERSIONS, PIN)


@pytest.mark.parametrize("name,version", [
    ("torch", "2.12.0+cu130"), ("nvidia-nccl-cu13", "2.29.7"), ("vllm", "0.30.0"),
    ("vllm", None),
])
def test_changed_or_missing_package_versions_reject_the_exception(name, version):
    versions = dict(VERSIONS)
    if version is None:
        del versions[name]
    else:
        versions[name] = version
    with pytest.raises(check.DependencyValidationError, match="Installed versions"):
        check.validate_pip_check(1, BUILD_DIAGNOSTIC, "", versions, PIN)


@pytest.mark.parametrize("override", [None, "", "nvidia-nccl-cu13==2.29.7\n",
                                      PIN + "nvidia-nccl-cu13==2.29.7\n"])
def test_missing_or_contradictory_vendor_pin_rejects_the_exception(override):
    with pytest.raises(check.DependencyValidationError, match="override pin"):
        check.validate_pip_check(1, BUILD_DIAGNOSTIC, "", VERSIONS, override)


@pytest.mark.parametrize("returncode,stderr", [(2, ""), (-9, ""), (1, "pip command failed\n")])
def test_subprocess_errors_cannot_be_accepted_as_dependency_conflicts(returncode, stderr):
    with pytest.raises(check.DependencyValidationError, match="unrecognized"):
        check.validate_pip_check(returncode, BUILD_DIAGNOSTIC, stderr, VERSIONS, PIN)


def mock_pip(monkeypatch, *, returncode=1, stdout=BUILD_DIAGNOSTIC, stderr=""):
    def run(command, **kwargs):
        assert command == [check.sys.executable, "-m", "pip", "check"]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(check.subprocess, "run", run)


def test_entrypoint_reads_actual_proof_and_preserves_original_diagnostic(
        monkeypatch, tmp_path, capsys):
    mock_pip(monkeypatch)
    override = tmp_path / "uv-overrides.txt"
    override.write_text(PIN)
    monkeypatch.setattr(check, "OVERRIDE_PATH", override)
    monkeypatch.setattr(check.importlib.metadata, "version", lambda name: VERSIONS[name])
    assert check.main() == 0
    output = capsys.readouterr()
    assert BUILD_DIAGNOSTIC in output.out
    assert "acknowledged the verified upstream" in output.out
    assert "ordinary pip check completed successfully" not in output.out


def test_entrypoint_missing_vendor_file_fails(monkeypatch, tmp_path, capsys):
    mock_pip(monkeypatch)
    monkeypatch.setattr(check, "OVERRIDE_PATH", tmp_path / "missing.txt")
    monkeypatch.setattr(check.importlib.metadata, "version", lambda name: VERSIONS[name])
    assert check.main() == 1
    assert "Dependency validation failed" in capsys.readouterr().err


def test_entrypoint_healthy_check_does_not_require_vendor_exception(monkeypatch, tmp_path, capsys):
    mock_pip(monkeypatch, returncode=0, stdout="No broken requirements found.\n")
    monkeypatch.setattr(check, "OVERRIDE_PATH", tmp_path / "no-vendor-file.txt")

    def unexpected_metadata_read(_name):
        raise AssertionError("Healthy checks should not read exception metadata")

    monkeypatch.setattr(check.importlib.metadata, "version", unexpected_metadata_read)
    assert check.main() == 0
    assert "ordinary pip check completed successfully" in capsys.readouterr().out


@pytest.mark.parametrize("nccl,expected_status", [("2.29.7", 1), ("2.30.7", 0)])
def test_entrypoint_enforces_vendor_stack_even_when_pip_check_reports_healthy(
        monkeypatch, tmp_path, capsys, nccl, expected_status):
    mock_pip(monkeypatch, returncode=0, stdout="No broken requirements found.\n")
    override = tmp_path / "uv-overrides.txt"
    override.write_text(PIN)
    versions = {**VERSIONS, "nvidia-nccl-cu13": nccl}
    monkeypatch.setattr(check, "OVERRIDE_PATH", override)
    monkeypatch.setattr(check.importlib.metadata, "version", lambda name: versions[name])
    assert check.main() == expected_status
    output = capsys.readouterr()
    assert "acknowledged the verified upstream" not in output.out
    if expected_status:
        assert "Installed versions do not match" in output.err
    else:
        assert "ordinary pip check completed successfully" in output.out


def test_entrypoint_missing_package_metadata_fails(monkeypatch, capsys):
    mock_pip(monkeypatch)

    def missing(name):
        raise check.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(check.importlib.metadata, "version", missing)
    assert check.main() == 1
    assert "Dependency validation failed" in capsys.readouterr().err


def test_entrypoint_preserves_unexpected_pip_exit_status(monkeypatch, capsys):
    mock_pip(monkeypatch, returncode=2, stdout="", stderr="pip internal error\n")
    assert check.main() == 2
    assert "pip internal error" in capsys.readouterr().err


def test_entrypoint_fails_if_pip_cannot_be_started(monkeypatch, capsys):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("python executable unavailable")

    monkeypatch.setattr(check.subprocess, "run", unavailable)
    assert check.main() == 1
    assert "Could not run pip check" in capsys.readouterr().err
