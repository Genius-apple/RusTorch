import os
import subprocess
import sys


def run_round(label, extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    env["RUST_TORCH_ROUND"] = label
    env.setdefault("RUST_TORCH_REPEAT", "3")
    env.setdefault("RUST_TORCH_MIN_SPEED_RATIO", "0.0")
    cmd = [sys.executable, "demo_visual/ci_regression.py"]
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"round failed: {label}")


def main():
    rounds = [
        ("round_7_auto_tensor", {}),
        ("round_8_profile_tensor", {"RUSTORCH_CPU_MATMUL_STRATEGY": "profile"}),
        (
            "round_9_profile_fused_grad",
            {"RUSTORCH_CPU_MATMUL_STRATEGY": "profile", "RUSTORCH_GRAD_PATH": "fused"},
        ),
        (
            "round_10_profile_fused_linear",
            {
                "RUSTORCH_CPU_MATMUL_STRATEGY": "profile",
                "RUSTORCH_LINEAR_FUSED": "1",
                "RUSTORCH_GRAD_PATH": "tensor",
            },
        ),
    ]
    for label, env in rounds:
        run_round(label, env)


if __name__ == "__main__":
    main()
