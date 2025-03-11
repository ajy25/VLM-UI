from pathlib import Path
import warnings
from functools import wraps
import os
from contextlib import redirect_stdout, redirect_stderr


def mute_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                return func(*args, **kwargs)

    return wrapper


def ignore_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


class VLM_Inference_Base:
    """Base model for standardized model structure"""

    def __init__(self, model_name: str, model_kwargs: dict) -> None:
        """Initialize model with given model name and kwargs"""
        self._model_name = model_name
        """Model name"""

        self._model_kwargs = model_kwargs
        """Model kwargs for model initialization"""

        self._chat_history: list[tuple[str, str]] = []
        """A list of tuples containing the speaker and their message; \
        e.g. [("human", "Hello!"), ("ai", "Hi!")]
        """

        self._image_path: Path = None
        """Path to the image"""

    def load_image(self, image_path: Path):
        """Load image from given path"""
        self._image_path = image_path
        # reset chat history
        self._chat_history = []

    def chat(self, prompt: str) -> str:
        """Chat with the model using given prompt"""
        return ""
