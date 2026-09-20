"""Local NVIDIA observations; never label a client GPU as the remote serving GPU."""
import asyncio
import csv
import io
import math

from contextbench.metrics.provenance import unavailable, utc_now

GPU_QUERY = "uuid,name,driver_version,pci.bus_id,utilization.gpu,memory.used,memory.total"


def parse_gpu(text: str) -> list[dict]:
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 7:
            raise ValueError("unexpected nvidia-smi column count")
        row = [v.strip() for v in row]
        device = dict(zip(("uuid", "name", "driver_version", "pci_bus_id"), row[:4]))
        device["unsupported"] = {}
        for key in ("uuid", "name", "driver_version", "pci_bus_id"):
            if device[key] in ("", "N/A", "[N/A]", "[Not Supported]"):
                device[key] = None
                device["unsupported"][key] = "nvidia-smi did not return this field"
        for name, raw in zip(("utilization_percent", "memory_used_mib", "memory_total_mib"), row[4:]):
            try:
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid measurement")
                device[name] = value
            except ValueError:
                device[name] = None
                device["unsupported"][name] = f"nvidia-smi returned {raw!r}"
        devices.append(device)
    return devices


async def sample_gpu() -> dict:
    source = f"nvidia-smi --query-gpu={GPU_QUERY} --format=csv,noheader,nounits"
    timestamp = utc_now()
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise
        if process.returncode:
            raise ValueError(stderr.decode(errors="replace") or f"exit code {process.returncode}")
        devices = parse_gpu(stdout.decode(errors="replace"))
        if not devices:
            raise ValueError("nvidia-smi returned no devices")
        return {"status": "available", "scope": "local_host", "source": source,
                "timestamp": timestamp, "reason": None, "devices": devices}
    except (OSError, ValueError, TimeoutError) as exc:
        return {**unavailable(source, f"{type(exc).__name__}: {exc}", "local_host"),
                "timestamp": timestamp, "devices": None}
