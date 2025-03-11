from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from ..base_model import VLM_Inference_Base


class CheXagent_Inference(VLM_Inference_Base):
    """Standard format class for CheXagent model inference."""

    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        huggingface_model_name: str = "StanfordAIMI/CheXagent-2-3b",
        dtype: torch.dtype = torch.float32,
        device_map: Literal[None, "auto"] = "auto",
        low_cpu_mem_usage: bool = True,
    ):
        """Initialize the CheXagent model inference class.

        Parameters
        ----------
        device : Literal["cpu", "cuda", "mpx"]
            The device to run the model on.

        huggingface_model_name : str, optional
            The Hugging Face model name to load, by default "StanfordAIMI/CheXagent-2-3b".

        dtype : torch.dtype, optional
            The data type to run the model on, by default torch.float32.

        device_map : Literal[None, "auto"], optional
            The device map to use for model loading, by default "auto".

        low_cpu_mem_usage : bool, optional
            Whether to use low CPU memory usage, by default True.
        """
        super().__init__(
            model_name="CheXagent",
            model_kwargs={
                "device": device,
                "huggingface_model_name": huggingface_model_name,
                "dtype": dtype,
                "device_map": device_map,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            },
        )
        self._dtype = dtype
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_name,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device)
        self._model = self._model.to(dtype)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            huggingface_model_name, trust_remote_code=True
        )
        self._formatted_chat_history = []

    def chat(self, prompt: str) -> str:
        """Chat with the model.

        Parameters
        ----------
        prompt : str
            The prompt/query to chat with the model.
        """
        if self._image_path is not None:
            paths = [self._image_path]
            paths = [str(path) for path in paths]
            query = self._tokenizer.from_list_format(
                [*[{"image": path} for path in paths], {"text": prompt}]
            )
        else:
            query = self._tokenizer.from_list_format([{"text": prompt}])
        if len(self._formatted_chat_history) > 0:
            conv = [*self._formatted_chat_history, {"from": "human", "value": query}]
        else:
            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
        input_ids = self._tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        output = self._model.generate(
            input_ids.to(self._device),
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=512,
        )[0]
        response = self._tokenizer.decode(output[input_ids.size(1) : -1])

        self._formatted_chat_history = conv + [{"from": "ai", "value": response}]
        self._chat_history.append(("human", prompt))
        self._chat_history.append(("ai", response))

        return response

    def load_image(self, image_path: Path):
        """Load image from given path"""
        super().load_image(image_path)

        # reset chat history
        self._formatted_chat_history = []
