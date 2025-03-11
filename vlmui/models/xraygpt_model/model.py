import sys
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "XrayGPT"))
sys.path.append(str(Path(__file__).parent / "XrayGPT" / "xraygpt"))


from XrayGPT.xraygpt.common.config import Config
from XrayGPT.xraygpt.common.dist_utils import get_rank
from XrayGPT.xraygpt.common.registry import registry
from XrayGPT.xraygpt.conversation.conversation import Chat, CONV_VISION
from XrayGPT.xraygpt.models import *
from XrayGPT.xraygpt.processors import *
from typing import Literal
from types import SimpleNamespace as Namespace
from ..base_model import VLM_Inference_Base


def load_pil_image(image_path):
    return Image.open(image_path).convert("RGB")


class XrayGPT_Inference(VLM_Inference_Base):
    """Standard format class for XrayGPT model inference."""

    def __init__(
        self, device: Literal["cpu", "cuda", "mps"] = "cpu", temperature: float = 0.1
    ):
        cfg = Config(
            Namespace(
                cfg_path=str(
                    Path(__file__).parent
                    / "XrayGPT"
                    / "eval_configs"
                    / "xraygpt_eval.yaml"
                ),
                gpu_id=0,
                options=None,
            )
        )
        print("XrayGPT model config:", model_config)
        model_config = cfg.model_cfg
        model_config.device_8bit = 0
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self._chat_model = Chat(model, vis_processor, device=device)
        self._chat_state = CONV_VISION.copy()
        self._formatted_chat_history = []
        self._temperature = temperature
        self._device = device
        self._image_list = []

    def chat(self, prompt: str) -> str:
        """Chat with the model.

        Parameters
        ----------
        prompt : str
            The prompt to chat with the model.
        """
        self._chat_model.ask(prompt, self._chat_state)
        llm_message = self._chat_model.answer(
            conv=self._chat_state,
            img_list=self._image_list,
            num_beams=1,
            temperature=self._temperature,
            max_new_tokens=300,
            max_length=2000,
        )[0]
        self._chat_history.append(("human", prompt))
        self._chat_history.append(("ai", llm_message))
        return llm_message

    def load_image(self, image_path):
        super().load_image(image_path)
        # reset the chat history
        self._chat_state = CONV_VISION.copy()
        self._image_list = []
        self._chat_model.upload_img(
            image=load_pil_image(image_path),
            conv=self._chat_state,
            img_list=self._image_list,
        )
