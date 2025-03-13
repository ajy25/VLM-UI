from vlmui.models.cxrllava_model import CXR_LLaVA_Inference
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()
imgs_dir = curr_dir / "test_images"

test_model = CXR_LLaVA_Inference(device="cpu")

# response = test_model.chat(
#     "Hi there! Tell me about yourself."
# )
# print("Model response:", response)

test_model.load_image(
    imgs_dir / "test0.png",
)

response = test_model.chat("Generate a radiology report for the image.")
print("Model response:", response)
