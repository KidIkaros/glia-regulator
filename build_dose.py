"""Build the dose-response kernel.

One builder, two targets:
  KERNEL_TARGET=gpu  -> mauricew/glia-dose-v1     (the real sweep, needs GPU quota)
  KERNEL_TARGET=cpu  -> mauricew/glia-dose-smoke  (a tiny end-to-end rehearsal that
                         costs no GPU quota)

The DEVICE default comes from the TARGET, not from the kernel's environment: a CPU
kernel that inherits DEVICE=cuda dies with "Torch not compiled with CUDA enabled",
which is exactly how the first CPU smoke failed.

    KERNEL_TARGET=cpu ALPHAS="0 1.0" N_PROBLEMS=1 MAX_NEW=64 python3 build_dose.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from kernel_build import build_kernel

TARGET = os.environ.get("KERNEL_TARGET", "gpu")
TARGETS = {
    "gpu": dict(kernel_id="mauricew/glia-dose-v1", kernel_dir="kernel_dose", gpu=True,
                device="cuda", alphas="0 0.5 1.0 1.5 2.0", problems="4", max_new="16384",
                deadline="5400"),
    "cpu": dict(kernel_id="mauricew/glia-dose-smoke", kernel_dir="kernel_dose_cpu",
                gpu=False, device="cpu", alphas="0 1.0", problems="1", max_new="64",
                deadline="900"),
}
config = TARGETS[TARGET]

build_kernel(
    kernel_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["kernel_dir"]),
    kernel_id=config["kernel_id"],
    files={
        "src/lever.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/lever.py"),
        "src/harness.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/harness.py"),
        "src/dose_response.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/dose_response.py"),
        "s1_pairs.json": os.path.join(os.path.dirname(os.path.abspath(__file__)), "s1_pairs.json"),
    },
    entry="src/dose_response.py",
    env={
        "ALPHAS": os.environ.get("ALPHAS", config["alphas"]),
        "N_PROBLEMS": os.environ.get("N_PROBLEMS", config["problems"]),
        "MAX_NEW": os.environ.get("MAX_NEW", config["max_new"]),
        "DEVICE": os.environ.get("DEVICE", config["device"]),
        "DEADLINE_S": os.environ.get("DEADLINE_S", config["deadline"]),
    },
    gpu=config["gpu"],
    label=f"{config['kernel_id']} ({TARGET})",
)
