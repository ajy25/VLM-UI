from pathlib import Path
import os
import subprocess
import sys
import re


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
    if not Path(curr_dir / "BiomedGPT-Base-Pretrained").exists():
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

    if not Path(curr_dir / "OFA").exists():
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
    # Read in setup.py
    with open(
        curr_dir / "OFA" / "transformers" / "setup.py", "r", encoding="utf-8"
    ) as f:
        content = f.read()

    # Modify the package name
    content = re.sub(r'name\s*=\s*"transformers"', 'name="transformers_old"', content)

    # Modify package_dir mapping
    content = re.sub(
        r'package_dir=\{"":\s*"src"\}',
        'package_dir={"transformers_old": "src/transformers"}',
        content,
    )

    # Modify packages to reflect the new name
    content = re.sub(
        r'packages=find_packages\("src"\)',
        'packages=["transformers_old"] + [f"transformers_old.{pkg}" for pkg in find_packages(where="src/transformers")]',
        content,
    )

    # Modify package_data
    content = re.sub(
        r'package_data=\{"transformers": \["py.typed"\]\}',
        'package_data={"transformers_old": ["py.typed"]}',
        content,
    )

    # Modify entry points
    content = re.sub(
        r"transformers-cli=transformers.commands.transformers_cli:main",
        "transformers-cli=transformers_old.commands.transformers_cli:main",
        content,
    )

    # Write back the modified content
    with open(
        curr_dir / "OFA" / "transformers" / "setup.py", "w", encoding="utf-8"
    ) as f:
        f.write(content)

    print(
        "setup.py has been successfully modified to rename transformers to transformers_old."
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
