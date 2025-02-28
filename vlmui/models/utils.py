import subprocess
import sys

ENV_NAME = "vlm-ui"


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
