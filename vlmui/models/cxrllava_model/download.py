from pathlib import Path
import os
import subprocess
import sys


def run_command(
    command: list[str],
    shell: bool = False,
):
    """Runs a shell command and handles errors."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)


curr_dir = Path(__file__).parent.resolve()


def download_CXRLLaVA():
    os.chdir(curr_dir)
    run_command(["pip", "install", "-r", str(curr_dir / "cxrllava-requirements.txt")])


if __name__ == "__main__":
    download_CXRLLaVA()
