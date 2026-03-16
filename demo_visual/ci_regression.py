import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_json_command(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    json_lines = []
    for line in lines:
        try:
            json_lines.append(json.loads(line))
        except Exception:
            pass
    if not json_lines:
        raise RuntimeError(f"no json output from {' '.join(cmd)}")
    return json_lines


def latest_finish(records, framework):
    candidates = [r for r in records if r.get("type") == "finish" and r.get("framework") == framework]
    if not candidates:
        raise RuntimeError(f"no finish record for {framework}")
    return candidates[-1]


def robust_center(values):
    if len(values) <= 2:
        return statistics.median(values)
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return med
    filtered = [v for v in values if abs(v - med) <= 3.5 * mad]
    if not filtered:
        return med
    return statistics.median(filtered)


def main():
    repeat = max(int(os.environ.get("RUST_TORCH_REPEAT", "3")), 1)
    torch_losses = []
    rust_losses = []
    torch_accs = []
    rust_accs = []
    torch_speeds = []
    rust_speeds = []

    rust_env = os.environ.copy()
    rust_env.setdefault("RUSTORCH_LINEAR_FUSED", "1")
    rust_env.setdefault("RUSTORCH_CPU_MATMUL_STRATEGY", "profile")
    rust_env.setdefault("RUSTORCH_CPU_REDUCTION_STRATEGY", "profile")
    rust_env.setdefault("RUSTORCH_CPU_ELEMWISE_STRATEGY", "profile")
    rust_env.setdefault("RUSTORCH_CPU_LAYERNORM_STRATEGY", "profile")
    rust_env.setdefault("RUSTORCH_FUSED_PIPELINE_STRATEGY", "profile")
    rust_env.setdefault("RUSTORCH_GRAD_PATH", "tensor")

    for _ in range(repeat):
        torch_records = run_json_command([sys.executable, "demo_visual/demo_benchmark.py"])
        rust_records = run_json_command(
            ["cargo", "run", "-p", "demo_visual", "--bin", "bench_alignment_ci", "--release"],
            env=rust_env,
        )
        torch_finish = latest_finish(torch_records, "PyTorch")
        rust_finish = latest_finish(rust_records, "RusTorch")
        torch_losses.append(max(float(torch_finish.get("final_loss", 0.0)), 1e-12))
        rust_losses.append(max(float(rust_finish.get("final_loss", 0.0)), 1e-12))
        torch_accs.append(float(torch_finish.get("final_accuracy", 0.0)))
        rust_accs.append(float(rust_finish.get("final_accuracy", 0.0)))
        torch_speeds.append(max(float(torch_finish.get("avg_speed", 0.0)), 1e-12))
        rust_speeds.append(max(float(rust_finish.get("avg_speed", 0.0)), 1e-12))

    torch_loss = robust_center(torch_losses)
    rust_loss = robust_center(rust_losses)
    torch_acc = robust_center(torch_accs)
    rust_acc = robust_center(rust_accs)
    torch_speed = robust_center(torch_speeds)
    rust_speed = robust_center(rust_speeds)

    log_gap = abs(math.log10(rust_loss) - math.log10(torch_loss))
    speed_ratio = rust_speed / torch_speed
    max_loss_ratio = float(os.environ.get("RUST_TORCH_MAX_LOSS_RATIO", "10.0"))
    min_speed_ratio = float(os.environ.get("RUST_TORCH_MIN_SPEED_RATIO", "0.15"))

    report = {
        "rust_final_loss": rust_loss,
        "torch_final_loss": torch_loss,
        "rust_final_acc": rust_acc,
        "torch_final_acc": torch_acc,
        "rust_avg_speed": rust_speed,
        "torch_avg_speed": torch_speed,
        "loss_log10_gap": log_gap,
        "speed_ratio_rust_over_torch": speed_ratio,
        "repeat": repeat,
        "torch_speed_samples": torch_speeds,
        "rust_speed_samples": rust_speeds,
    }
    print(json.dumps(report, indent=2))

    history_path = Path("demo_visual/speed_ratio_history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    round_label = os.environ.get("RUST_TORCH_ROUND", "auto")
    if not history_path.exists():
        history_path.write_text(
            "timestamp,round,speed_ratio,rust_avg_speed,torch_avg_speed,rust_final_loss,torch_final_loss\n",
            encoding="utf-8",
        )
    with history_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{int(time.time())},{round_label},{speed_ratio:.9f},{rust_speed:.6f},{torch_speed:.6f},{rust_loss:.12e},{torch_loss:.12e}\n"
        )

    ok = True
    if rust_acc < 0.999:
        ok = False
        print("FAIL: rust accuracy < 99.9%", file=sys.stderr)
    if rust_loss > torch_loss * max_loss_ratio:
        ok = False
        print(f"FAIL: rust loss is worse than torch by > {max_loss_ratio}x", file=sys.stderr)
    if speed_ratio < min_speed_ratio:
        ok = False
        print(f"FAIL: speed ratio rust/torch < {min_speed_ratio}", file=sys.stderr)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
