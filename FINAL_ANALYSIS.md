# Final Analysis

**Status: BLOCKED**

No validated benchmark outputs are available. The pilot is blocked because this environment has no `nvidia-smi` GPU runtime and no reachable vLLM endpoint. No performance, quality, cache, or variance conclusions are reported.

Run `scripts/run_pilot.py` on the configured GPU/vLLM host. After `results/pilot/pilot_validation.json` reports `PASS`, run `scripts/run_full_benchmark.py`, then regenerate this document with `scripts/final_analysis.py`.
