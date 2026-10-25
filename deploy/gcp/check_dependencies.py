"""Check all dependencies while recognizing one documented vLLM image override.

vLLM 0.29.0 intentionally replaces torch's older NCCL pin for DeepEPv2:
https://github.com/vllm-project/vllm/blob/v0.29.0/docker/Dockerfile#L854
The exception is valid only for the CUDA 12.9 profile and vendor override file.
Every other pip-check failure remains fatal; CUDA runtime checks belong to the
real GPU smoke task, not to this metadata audit.
"""

import importlib.metadata
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

EXPECTED_CONFLICT = (
    'torch 2.13.0+cu129 has requirement nvidia-nccl-cu12==2.29.7; '
    'platform_system == "Linux", but you have nvidia-nccl-cu12 2.30.7.'
)
EXPECTED_VERSIONS = {
    "torch": "2.13.0+cu129",
    "nvidia-nccl-cu12": "2.30.7",
    "vllm": "0.29.0",
}
OVERRIDE_PIN = "nvidia-nccl-cu12==2.30.7"
OVERRIDE_PATH = Path("/etc/uv-overrides.txt")


class DependencyValidationError(RuntimeError):
    pass


def validate_vendor_override(versions: Mapping[str, str] | None,
                             override_text: str | None):
    if versions is None or any(versions.get(name) != version
                               for name, version in EXPECTED_VERSIONS.items()):
        raise DependencyValidationError("Installed versions do not match the documented override")
    pins = [line.split("#", 1)[0].strip() for line in (override_text or "").splitlines()]
    nccl_pins = [line for line in pins if line.startswith("nvidia-nccl-cu")]
    if nccl_pins != [OVERRIDE_PIN]:
        raise DependencyValidationError("The upstream NCCL override pin is missing or changed")


def validate_pip_check(returncode: int, stdout: str, stderr: str,
                       versions: Mapping[str, str] | None = None,
                       override_text: str | None = None) -> bool:
    """Return whether the verified vendor exception was used; fail closed otherwise."""
    if returncode == 0:
        # A downgrade could satisfy torch while violating the image's DeepEP
        # contract. Enforce vendor proof even when pip reports healthy metadata.
        if versions is not None or override_text is not None:
            validate_vendor_override(versions, override_text)
        return False
    if returncode != 1 or stderr.strip() or stdout.strip() != EXPECTED_CONFLICT:
        raise DependencyValidationError("pip check reported an unrecognized dependency failure")
    validate_vendor_override(versions, override_text)
    return True


def main() -> int:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True, check=False)
    except OSError as exc:
        print(f"Could not run pip check: {type(exc).__name__}", file=sys.stderr)
        return 1
    # Preserve the original diagnostic, including the acknowledged vendor conflict.
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    versions, override_text = None, None
    try:
        if OVERRIDE_PATH.is_file() or (
                result.returncode == 1 and result.stdout.strip() == EXPECTED_CONFLICT):
            versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
            override_text = OVERRIDE_PATH.read_text(encoding="utf-8")
        used_override = validate_pip_check(result.returncode, result.stdout, result.stderr,
                                          versions, override_text)
    except (DependencyValidationError, importlib.metadata.PackageNotFoundError, OSError) as exc:
        print(f"Dependency validation failed: {exc}", file=sys.stderr)
        return result.returncode if result.returncode > 0 else 1
    if used_override:
        print("Dependency validation acknowledged the verified upstream vLLM 0.29.0 "
              "NCCL 2.30.7 override. Ordinary pip check reports the conflict above; "
              "no other dependency conflicts were accepted.")
    else:
        print("Dependency validation: ordinary pip check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
