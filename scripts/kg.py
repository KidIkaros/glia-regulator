"""Thin Python wrapper around the Kaggle API (the CLI's auth path is broken when the
OAuth access token expires; the library path still works).

Usage:
    python3 scripts/kg.py quota
    python3 scripts/kg.py status <slug>
    python3 scripts/kg.py logs <slug> [-f]
    python3 scripts/kg.py push <dir>
    python3 scripts/kg.py delete <slug>
    python3 scripts/kg.py list
"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "quota":
        print(api.benchmarks_quota())
    elif cmd == "status":
        print(api.kernel_status(rest[0]))
    elif cmd == "logs":
        flags = {}
        for a in rest[1:]:
            if a == "-f":
                flags["file"] = "/tmp/kagglelogs.txt"
        print(api.kernel_output(rest[0], **flags))
    elif cmd == "push":
        api.kernels_push(rest[0])
    elif cmd == "delete":
        api.kernels_delete(rest[0])
    elif cmd == "list":
        for k in api.kernels_list():
            print(k.ref, k.title, k.last_run_time)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
