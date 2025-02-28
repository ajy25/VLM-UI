from typing import Literal
from pathlib import Path
from .CheXagent import CheXagent_Inference


class ModelIntegrator:
    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):
        self._name_to_model = {
            "CheXagent": CheXagent_Inference(device=device),
        }
        self._current_model_name = sorted(self._name_to_model.keys())[0]
        self._current_model = self._name_to_model[self._current_model_name]
        self._current_image_path = None

    def set_model(self, model_name: str):
        self._current_model_name = model_name
        self._current_model = self._name_to_model[model_name]
        if self._current_image_path is not None:
            self._current_model.load_image(self._current_image_path)

    def set_image_path(self, image_path: Path):
        self._current_image_path = image_path
        self._current_model.load_image(image_path)

    def get_current_model(self):
        if self._current_model is None:
            raise ValueError("No model is currently set.")
        return self._current_model

    def get_current_model_name(self):
        return self._current_model_name

    def get_model_by_name(self, model_name: str):
        return self._name_to_model[model_name]
