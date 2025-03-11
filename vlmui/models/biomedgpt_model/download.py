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


def download_BiomedGPT():
    os.chdir(curr_dir)
    if not Path("BiomedGPT-Base-Pretrained").exists():
        print("Downloading BiomedGPT-Base-Pretrained...")
        run_command(
            [
                "git",
                "clone",
                "https://huggingface.co/PanaceaAI/BiomedGPT-Base-Pretrained",
            ]
        )
        os.chdir(curr_dir / "BiomedGPT-Base-Pretrained")
        run_command(["git", "lfs", "install"])
        print("Pulling BiomedGPT-Base-Pretrained...")
        run_command(["git", "lfs", "pull"])
        os.chdir(curr_dir)
    else:
        print("BiomedGPT-Base-Pretrained already exists.")

    if not Path("OFA").exists():
        print("Downloading OFA...")
        run_command(
            [
                "git",
                "clone",
                "--single-branch",
                "--branch",
                "feature/add_transformers",
                "https://github.com/OFA-Sys/OFA.git",
            ]
        )
        run_command(
            [
                "pip",
                "install",
                str(curr_dir / "OFA" / "transformers"),
            ]
        )
    run_command(
        [
            "pip",
            "install",
            "-r",
            "biomedgpt-requirements.txt",
        ]
    )


if __name__ == "__main__":
    download_BiomedGPT()
