from pathlib import Path
import os
from ..utils import run_command, ENV_NAME

curr_dir = Path(__file__).parent.resolve()


def download_CheXagent():
    os.chdir(curr_dir)
    run_command(["pip", "install", "-r", str(curr_dir / "requirements.txt")])
