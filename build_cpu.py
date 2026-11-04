"""Build the CPU-only kernels for the lever's horizon and trajectory measurements.

Both drivers are the same experiment family (how faithfully does the runtime lever
reproduce the weight edit?), so one builder with the entry script chosen by env:

    KERNEL_ENTRY=src/cpu_trajectory.py python3 build_cpu.py
    KERNEL_ENTRY=src/cpu_horizon.py    python3 build_cpu.py

CPU kernels draw no GPU quota, which matters while the GPU meter is spent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from kernel_build import build_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.environ.get("KERNEL_ENTRY", "src/cpu_trajectory.py")
SUFFIX = "traj" if "trajectory" in ENTRY else "horizon"

build_kernel(
    kernel_dir=os.path.join(HERE, "kernel_cpu"),
    kernel_id=f"mauricew/glia-cpu-{SUFFIX}",
    files={
        "src/lever.py": os.path.join(HERE, "src/lever.py"),
        "src/harness.py": os.path.join(HERE, "src/harness.py"),
        ENTRY: os.path.join(HERE, ENTRY),
        "s1_pairs.json": os.path.join(HERE, "s1_pairs.json"),
    },
    entry=ENTRY,
    env={
        "ALPHAS": os.environ.get("ALPHAS", "1.0 1.5 2.0"),
        "ROLLOUT": os.environ.get("ROLLOUT", "1024"),
        "CHECKPOINTS": os.environ.get("CHECKPOINTS", "8"),
        "DEADLINE_S": os.environ.get("DEADLINE_S", "5400"),
        "N_THREADS": os.environ.get("N_THREADS", "4"),
    },
    gpu=False,
    label=f"glia-cpu-{SUFFIX}",
)
