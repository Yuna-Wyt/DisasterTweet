from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from peft import PeftModel, config

model_path = "/mnt/userdata/model/llama3"
lora_path = "/mnt/userdata/result/mnt/userdata/result/save"

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#加载模型
model = AutoModelForCausalLM.from_pretrained(model_path,device_map ='auto',torch_dtype=torch.bfloat16)

#加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "你是谁？"
message = [
    {"role":"system", "content":"现在你要扮演皇帝身边的女人--甄嬛"},
    {"role":"user", "content":prompt}
]

input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([input_text], return_tensors='pt',padding=True).to('cuda')

generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    max_new_tokens = 512,
    do_sample=True,
    top_p=0.9,
    temperature=0.5,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.encode('<|eot_id|>')[0]
)
generated_ids = [output_ids[len(input_ids):]for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)