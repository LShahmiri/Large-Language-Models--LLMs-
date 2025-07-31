

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/phi-3-mini-4k-instruct"  # ✅ correct name with dash

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt='Write an email aplpogizeing to sarah for the the broken dish and explain that how it happened.<|assistant|>'

#Tokenize the input prompt
input_ids=tokenizer(prompt,return_tensors='pt').input_ids.to('cuda')
#Generate the text
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=200,
    use_cache=False ) # ✅ This avoids DynamicCache

#Print the output
print(tokenizer.decode(generation_output[0]))

print (input_ids)

for id in input_ids[0]:
  print(tokenizer.decode(id))

print(generation_output)

for id in generation_output[0]:
  print(tokenizer.decode(id))

print(tokenizer.decode(373))

