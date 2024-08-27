import json
import os
import argparse

import numpy as np
import torch
import wandb
import tqdm
from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset


def write_to_file(datasets,file_path):
    content = '\n'.join(datasets)
    with open(file_path, 'a') as f:
        f.write(content + '\n')

def get_dataset(data_path):
    dataset = load_dataset("csv", data_files=data_path, split="train")
    print(f"dataset:\n{dataset}")

    def text_preprocess_function(examples):
        prompt = f"""
### Instruction:
    You will receive a set of inputs that include **Keyword**, **Location**, and **Text**. Your task is to determine whether the Tweet is about a real disaster or not. Consider the following aspects:
    - **Keyword**: Evaluate whether the keyword is associated with disaster-related events, effects, or implications. Note that the keyword may be blank.
    - **Location**: Assess if the location mentioned has any known history or characteristics that could relate to disasters. Note that the location may be blank.
    - **Text**: Analyze the text to identify any descriptions, implications, or hints of a disaster scenario. This could include direct mentions or subtle references to disastrous events.
        
Learn the first 8 examples and predict the output of the final input.The output must be one word, which is 'Yes' or 'No'.

### Input:
    - **Keyword**: snowstorm
    - **Location**: South, USA
    - **Text**: Sassy city girl country hunk stranded in Smoky Mountain snowstorm #AoMS http://t.co/nkKcTttsD9 #ibooklove #bookboost
### Output: Yes

### Input:
    - **Keyword**: hostage
    - **Location**: Roaming around the world
    - **Text**: Islamic State group in Egypt threatens to kill Croat hostage http://t.co/NzIfztCUGL
### Output: Yes

### Input:
    - **Keyword**: armageddon
    - **Location**: Worldwide
    - **Text**: God's Kingdom (Heavenly Gov't) will rule over all people on the earth after Armageddon.  http://t.co/8HGcBXUkz0  http://t.co/4kopkCyvTt
### Output: No

### Input:
    - **Keyword**: airplane accident
    - **Location**: 92
    - **Text**: #OMG! I don't believe this. #RIP bro#AirPlane #Accident #JetEngine #TurboJet #Boing #G90 http://t.co/KXxnSZp6nk
### Output: Yes

### Input:
    - **Keyword**: blew up
    - **Location**: Atlanta
    - **Text**: @mfalcon21 go look. Just blew it up w atomic bomb.
### Output: No

### Input:
    - **Keyword**: fatality
    - **Location**: Hueco Mundo
    - **Text**: I liked a @YouTube video from @vgbootcamp http://t.co/yi3OiVK2X4 S@X 109 - sN | vaBengal (ZSS) Vs. SWS | Fatality (Captain Falcon)
### Output: No

### Input:
    - **Keyword**: engulfed
    - **Location**: London
    - **Text**: 'Tube strike live: Latest travel updates as London engulfed in chaos' &lt;- genuine baffling Telegraph headline
### Output: Yes

### Input:
    - **Keyword**: cliff fall
    - **Location**: None
    - **Text**: Beat the #heat. Today only Kill Cliff Free Fall $2. Pick up a #cold drink today after the #tough #crossfit... http://t.co/QaMwoJYahq
### Output: No

### Input:
    - **Keyword**: {examples['keyword']}
    - **Location**: {examples['location']}
    - **Text**: {examples['text']}
### Output:
"""

        # message = [
        #     {"role":"user", "content":prompt},
        # ]
        input_text = prompt

        targets = examples["target"]

        return {"inputs_text":input_text,"targets":targets}

    dataset = dataset.map(text_preprocess_function, remove_columns=dataset.column_names)
    return dataset

def eval(model, tokenizer , data_path):
    dataset = get_dataset(data_path, )
    print(data_path)
    acc,err,num,step = 0,0,0,0
    eval = np.array([[0,0],[0,0]])
    result = {}
    error = []
    model.eval()

    for item in tqdm.tqdm(dataset,total=len(dataset)):
        input_text = item["inputs_text"]

        input_ids = tokenizer(input_text,return_tensors="pt").input_ids
        output_ids = model.generate(
            input_ids = input_ids,
            max_new_tokens = 10,
            pad_token_id = tokenizer.eos_token_id
        )
        output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, output_ids)
        ]

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(output_text)

        output_label = 0
        if "Yes" in output_text and "No" not in output_text:
            output_label = 1
        elif "No" in output_text and "Yes" not in output_text:
            output_label = 0
        else:
            err += 1
            error.append(json.dumps(item))
        print(output_label)
        print(item['targets'])
        num += 1

        if output_label == 0 or output_label == 1:
            eval[output_label][item['targets']] += 1
        else:
            err += 1
            error.append(json.dumps(item))
        step += 1
        epsilon = 1e-7
        if step % 10 == 0:
            result["accuracy"] = eval
            result["F1"] = 2 * eval[1][1] / (2 * eval[1][1] + eval[0][1] + eval[1][0] + epsilon)
            result["error_num"] = err
            result["acc"] = (eval[0][0] + eval[1][1]) / num
            print(result)
            error = []
            # result_serializable = {
            #     key : (value.tolist() if isinstance(value, np.ndarray) else value) for key,value in result.items()
            # }
            # write_to_file(result_serializable,f"{data_path}/result.json")
    # wandb.log({
    #     "acc": result["acc"],
    #     "F1":result["F1"],
    #     "err":result["error_num"],
    #     "eval1_1":eval[1][1],
    #     "eval0_0":eval[0][0],
    # })
    # wandb.finish()
# os.environ["CUDA_VISIBLE_DEVICES"] = " "
# os.environ["WANDB_PROJECT"] = " "
# os.environ["WANDB_API_KEY"] = " "

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/userdata/result/mnt/userdata/result/dataset/test_set.csv",
        help=""
    )
    args = parser.parse_known_args()[0]
    # wandb.init(project="NLP_DisasterTweets")
    print(args)
    data_path = args.data_path

    model_path = "/mnt/userdata/model/llama3"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto")
    # lora_path  = "/mnt/userdata/result/mnt/userdata/result/save"
    # model = PeftModel.from_pretrained(model,lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    eval(model=model, tokenizer=tokenizer,data_path=data_path)
