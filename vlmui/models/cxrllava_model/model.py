from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel
from typing import Literal
from ..base_model import VLM_Inference_Base


class CXR_LLaVA_Inference(VLM_Inference_Base):
    """Standard format class for CXR-LLaVA model inference."""

    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        huggingface_model_name: str = "ECOFRI/CXR-LLAVA-v2",
        device_map: Literal[None, "auto"] = "auto",
        low_cpu_mem_usage: bool = True,
    ):
        """Initialize the CXR-LLaVA model inference class.

        Parameters
        ----------
        device : Literal["cpu", "cuda", "mpx"]
            The device to run the model on.

        huggingface_model_name : str, optional
            The Hugging Face model name to load, by default "ECOFRI/CXR-LLAVA-v2".

        device_map : Literal[None, "auto"], optional
            The device map to use for model loading, by default "auto".

        low_cpu_mem_usage : bool, optional
            Whether to use low CPU memory usage, by default True.
        """
        super().__init__(
            model_name="CXR-LLaVA",
            model_kwargs={
                "device": device,
                "huggingface_model_name": huggingface_model_name,
                "device_map": device_map,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            },
        )
        self._device = device
        self._model = AutoModel.from_pretrained(
            huggingface_model_name,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device)
        self._model.eval()
        self._image = None
        self._formatted_chat_history = []

    def chat(self, prompt: str) -> str:
        """Chat with the model.

        Parameters
        ----------
        prompt : str
            The prompt to chat with the model.
        """
        if self._image is None:
            return "Please provide an image first."
        if prompt == "":
            return self._model.write_radiologic_report(self._image)

        if len(self._formatted_chat_history) == 0:
            self._formatted_chat_history = [
                {
                    "role": "system",
                    "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides.",
                },
                {"role": "user", "content": f"<image>\n{prompt}\n"},
            ]
        else:
            self._formatted_chat_history.append(
                {"role": "user", "content": f"<image>\n{prompt}\n"}
            )

        response = self._model.generate_cxr_repsonse(
            chat=self._formatted_chat_history,
            pil_image=self._image,
            temperature=0,
            top_p=1,
        )
        self._chat_history.append(
            (
                ("human", prompt),
                ("ai", response),
            )
        )
        self._formatted_chat_history.append({"role": "assistant", "content": response})

    def load_image(self, image_path: Path):
        """Load an image for the model to process.

        Parameters
        ----------
        image_path : Path
            The path to the image to load.
        """
        super().load_image(image_path)
        self._image = Image.open(image_path)

        # reset chat history
        self._formatted_chat_history = []
