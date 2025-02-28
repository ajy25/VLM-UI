from pathlib import Path
import os
import re
import subprocess
import sys
from ..utils import run_command, ENV_NAME

curr_dir = Path(__file__).parent.resolve()


INPUT_FILE = curr_dir / "XrayGPT" / "xraygpt_requirements.txt"
OUTPUT_FILE = curr_dir / "XrayGPT" / "xraygpt_requirements_fixed.txt"


ignore = ["decord", "mkl-fft", "mkl-random"]
strip_version = []


def get_latest_version(package_name):
    """Retrieve the latest version of a package from PyPI."""
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Extract the latest version
        match = re.search(r"\(([\d.]+)\)", output)
        return match.group(1) if match else None
    except Exception as e:
        print(f"Warning: Could not fetch version for {package_name}. Error: {e}")
        return None


def fix_requirements(input_file, output_file):
    """Replace file-based dependencies with the latest PyPI versions."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:

            if line.split("==")[0] in ignore:
                print(f"Removing deprecated package: {line.strip()}")
                continue
            if line.split("==")[0] in strip_version:
                print(f"Replacing: {line.strip()} → {line.split('==')[0]}")
                outfile.write(line.split("==")[0] + "\n")
                continue

            match = re.match(r"([\w\-]+) @ file://.*", line.strip())
            if match:
                package = match.group(1)
                latest_version = get_latest_version(package)
                if latest_version:
                    new_line = f"{package}=={latest_version}\n"
                    print(f"Replacing: {line.strip()} → {new_line.strip()}")
                    outfile.write(new_line)
                else:
                    newline = f"{package}\n"
                    print(f"Warning: Could not find version for {package}.")
                    print(f"Replacing: {line.strip()} → {newline.strip()}")
                    outfile.write(newline)
            else:
                outfile.write(line)

    print(f"\n✅ Cleaned dependencies saved to {output_file}")


def download_xraygpt():
    os.chdir(curr_dir)
    run_command(["rm", "-rf", "XrayGPT"])
    run_command(["git", "clone", "https://github.com/mbzuai-oryx/XrayGPT.git"])
    assert (curr_dir / "XrayGPT").exists(), "XrayGPT repo not found."
    fix_requirements(INPUT_FILE, OUTPUT_FILE)
    run_command(
        [
            "pip",
            "install",
            "-r",
            str(curr_dir / "XrayGPT" / "xraygpt_requirements_fixed.txt"),
        ]
    )
