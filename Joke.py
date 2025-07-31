from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",  # Use GPU
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt
prompt = "Create a funny joke about chickens."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    use_cache=False  # 
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))