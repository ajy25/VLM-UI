from vlmui.models.biomedgpt_model import BiomedGPT_Inference
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()
imgs_dir = curr_dir / "test_images"

test_model = BiomedGPT_Inference(device="cpu")

test_model.load_image(
    imgs_dir / "test0.png",
)

response = test_model.chat("Describe the image.")
print("Model response:", response)
