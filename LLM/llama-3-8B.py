from transformers import AutoTokenizer,AutoModelForCausalLM
device = "cuda"

model_id = "/mnt/userdata/model/llama3"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map ="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

message = [
    {"role":"system","content":"You are a pirate chatbot who always responds in pirate speak!"},#系统角色设定
    {"role":"user","content":"Who are you?"}#用户角色输入提示
]

text = tokenizer.apply_chat_template(
    message,
    tokenize = False,
    add_generation_prompt=True
)
print(type(text))
print(text)

model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)

print(model_inputs)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=256
)
print(generated_ids)
generated_ids = [output_ids[len(input_ids):]for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

print(generated_ids)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
