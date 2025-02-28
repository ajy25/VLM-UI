import io
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# step 1: Setup constant
model_name = "StanfordAIMI/CheXagent-2-3b"
dtype = torch.bfloat16
device = "cpu"

# step 2: Load Processor and Model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map=None,  # Disable offloading
    low_cpu_mem_usage=False,  # Ensure parameters load directly to memory
).to(device)
model = model.to(dtype)
model.eval()


curr_dir = Path(__file__).parent.resolve()
imgs_dir = curr_dir.parent.parent.parent / "test_images"

paths = [
    # imgs_dir / "test0.png",
    imgs_dir
    / "test1.png",
]
paths = [str(path) for path in paths]

prompt = "Generate a radiology report for the image."


# step 3: Inference
query = tokenizer.from_list_format(
    [*[{"image": path} for path in paths], {"text": prompt}]
)
conv = [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": query},
]
input_ids = tokenizer.apply_chat_template(
    conv, add_generation_prompt=True, return_tensors="pt"
)
output = model.generate(
    input_ids.to(device),
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_p=1.0,
    use_cache=True,
    max_new_tokens=512,
)[0]
response = tokenizer.decode(output[input_ids.size(1) : -1])


print(response)
