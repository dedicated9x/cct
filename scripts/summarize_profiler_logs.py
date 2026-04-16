#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List


def _first_float_after(line: str) -> Optional[float]:
    # Looks for typical torch profiler columns: ncalls tottime percall cumtime percall ...
    # Example:
    #   500    0.020    0.000    2.278    0.005 arch.py:109(forward)
    m = re.search(r"\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+", line)
    if not m:
        return None
    return float(m.group(4))  # cumtime


def summarize_one(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="ignore").splitlines()

    def find_cumtime(pattern: str) -> Optional[float]:
        # Return the first matching cumtime.
        rx = re.compile(pattern)
        for line in text:
            if rx.search(line):
                return _first_float_after(line)
        return None

    # GPU line near the top
    gpu_line = None
    for line in text[:200]:
        if "GPU available" in line:
            gpu_line = line.strip()
            break

    forward_cum = find_cumtime(r"arch\.py:\d+\(forward\)|arch\.py:109\(forward\)")
    loss_cum = find_cumtime(r"metrics\.py:\d+\(bcewithlogits_multilabel\)|metrics\.py:5\(bcewithlogits_multilabel\)")
    training_step_cum = find_cumtime(r"module\.py:\d+\(training_step\)|module\.py:58\(training_step\)")

    return {
        "path": str(path),
        "gpu_line": gpu_line,
        "forward_cum_s": forward_cum,
        "loss_cum_s": loss_cum,
        "training_step_cum_s": training_step_cum,
    }


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: summarize_profiler_logs.py <log1> [log2 ...]")
        return 2

    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"Missing: {path}")
            continue
        s = summarize_one(path)
        print(f"\n== {path.name} ==")
        print(f"GPU: {s['gpu_line']}")
        if s["forward_cum_s"] is not None:
            print(f"cumtime forward: {s['forward_cum_s']:.3f}s")
        else:
            print("cumtime forward: (not found)")
        if s["loss_cum_s"] is not None:
            print(f"cumtime bcewithlogits_multilabel: {s['loss_cum_s']:.3f}s")
        else:
            print("cumtime bcewithlogits_multilabel: (not found)")
        if s["training_step_cum_s"] is not None:
            print(f"cumtime training_step: {s['training_step_cum_s']:.3f}s")
        else:
            print("cumtime training_step: (not found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

