from pathlib import Path
from vlmui.models.xraygpt_model.model import XrayGPT_Inference


curr_dir = Path(__file__).parent.parent.parent.parent

if __name__ == "__main__":
    print(curr_dir)
    model = XrayGPT_Inference(device="cpu")
    model.load_image(curr_dir / "test_images" / "test0.png")
    print(model.chat("What insights can you provide about the X-ray image?"))
