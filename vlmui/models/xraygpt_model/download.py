from pathlib import Path
import os
import subprocess
import sys
import yaml
import gdown


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


def update_ckpt_path(yaml_file, new_ckpt_path):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    config["model"]["ckpt"] = new_ckpt_path

    with open(yaml_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    print(f"Updated ckpt path to: {new_ckpt_path}")


def download_XrayGPT():
    os.chdir(curr_dir)

    if not (curr_dir / "XrayGPT").exists():
        run_command(["git", "clone", "https://github.com/mbzuai-oryx/XrayGPT.git"])
        assert (curr_dir / "XrayGPT").exists(), "XrayGPT download failed."
    else:
        print("Cloned repository `XrayGPT` already exists.")

    print("Installing XrayGPT dependencies...")
    run_command(["pip", "install", "-r", str(curr_dir / "xraygpt-requirements.txt")])

    # download the pretrained path
    print("Downloading pretrained XrayGPT model...")
    url = "https://drive.google.com/file/d/1h50ZNMyryJwju126gYe2X1Q6S_JLGs7Z/view?usp=drive_link"
    (curr_dir / "XrayGPT" / "checkpoints").mkdir(exist_ok=True)
    output = str(curr_dir / "XrayGPT" / "checkpoints" / "XrayGPT.pth")
    gdown.download(url, output, quiet=False)

    # need to update the yaml ckpt
    print("Updating the XrayGPT yaml file...")
    update_ckpt_path(
        curr_dir / "XrayGPT" / "eval_configs" / "xraygpt_eval.yaml",
        str(curr_dir / "XrayGPT" / "checkpoints" / "XrayGPT.pth"),
    )

    print("XrayGPT model downloaded and ready to use.")


if __name__ == "__main__":
    download_XrayGPT()
