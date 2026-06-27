#!/usr/bin/env python3
"""
Install Python API dependencies into /scratch/bin.

Run with:  python3 install_dependencies.py
"""

import subprocess
import sys


PACKAGES = [
    "websockets",
    "ipython",
    "numpy",
    "trimesh",
    "rtree",
]


def main():
    target = "/scratch/bin"
    pip = [sys.executable, "-m", "pip", "install", "--target", target]

    print(f"Using Python: {sys.executable}")
    print(f"Installing into: {target}")

    print("Installing prerequisite: wheel\n")
    pre = subprocess.run([sys.executable, "-m", "pip", "install", "--user", "wheel"])
    if pre.returncode != 0:
        print("\nFailed to install wheel.", file=sys.stderr)
        sys.exit(pre.returncode)

    print(f"\nInstalling: {', '.join(PACKAGES)}\n")

    result = subprocess.run(pip + PACKAGES)

    if result.returncode != 0:
        print("\nInstallation failed.", file=sys.stderr)
        sys.exit(result.returncode)

    print("\nAll dependencies installed.")
    print(f"If needed, add the target to PYTHONPATH:")
    print(f"  export PYTHONPATH=\"{target}:$PYTHONPATH\"")


if __name__ == "__main__":
    main()
