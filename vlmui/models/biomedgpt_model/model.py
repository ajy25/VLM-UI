from transformers import OFATokenizer, OFAModel
import re
import torch
from PIL import Image
from torchvision import transforms
from typing import Literal
from pathlib import Path
from ..base_model import VLM_Inference_Base

curr_dir = Path(__file__).parent.resolve()


class BiomedGPT_Inference(VLM_Inference_Base):
    """Standard format class for BiomedGPT model inference."""

    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        huggingface_model_name: str = "BiomedGPT-Base-Pretrained",
        device_map: Literal[None, "auto"] = "auto",
        low_cpu_mem_usage: bool = True,
    ):
        """Initialize the BiomedGPT model inference class.

        Parameters
        ----------
        device : Literal["cpu", "cuda", "mpx"]
            The device to run the model on.

        huggingface_model_name : str, optional
            The Hugging Face model name to load, by default "BiomedGPT-Base-Pretrained".

        device_map : Literal[None, "auto"], optional
            The device map to use for model loading, by default "auto".

        low_cpu_mem_usage : bool, optional
            Whether to use low CPU memory usage, by default True.
        """
        super().__init__(
            model_name="BiomedGPT",
            model_kwargs={
                "device": device,
                "huggingface_model_name": huggingface_model_name,
            },
        )
        self._model_name = huggingface_model_name
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self._patch_resize_transform = transforms.Compose(
            [
                lambda image: image.convert("RGB"),
                transforms.Resize(
                    (resolution, resolution), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self._device = device
        self._model = OFAModel.from_pretrained(
            str(curr_dir / self._model_name),
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device)
        self._model.eval()
        self._tokenizer = OFATokenizer.from_pretrained(str(curr_dir / self._model_name))

    def chat(
        self,
        prompt: str,
    ) -> str:
        """Chat with the model.

        Parameters
        ----------
        prompt : str
            The prompt to chat with the model.
        """
        inputs = self._tokenizer([prompt], return_tensors="pt").input_ids
        patch_img = self._patch_resize_transform(self._img).unsqueeze(0)
        gen = self._model.generate(
            inputs,
            patch_images=patch_img,
            num_beams=5,
            no_repeat_ngram_size=3,
            max_length=16,
        )
        results = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
        result = results[0]
        result = re.sub(r"[^\w\s]", "", result).strip()
        return result

    def load_image(self, image_path: Path):
        """Load image from given path"""
        super().load_image(image_path)
        self._img = Image.open(image_path)
