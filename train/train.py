import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DefaultDataCollator, AutoModelForCausalLM, \
    TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_path = "/mnt/userdata/model/llama3"
train_path = "/mnt/userdata/result/mnt/userdata/result/dataset/train_set.csv"
save_path = "/mnt/userdata/result/mnt/userdata/result/save"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def get_dataset(data_path):
    dataset = load_dataset("csv",data_files=data_path,split="train")

    def text_preprocess_function(examples,max_length=950):
        #对单个样本进行处理
        prompt = f"""### Instruction:
         You will receive a set of inputs that include **Keyword**, **Location**, and **Text**. Your task is to determine whether the Tweet is about a real disaster or not. Consider the following aspects:
         - **Keyword**: Evaluate whether the keyword is associated with disaster-related events, effects, or implications. Note that the keyword may be blank.
         - **Location**: Assess if the location mentioned has any known history or characteristics that could relate to disasters. Note that the location may be blank.
         - **Text**: Analyze the text to identify any descriptions, implications, or hints of a disaster scenario. This could include direct mentions or subtle references to disastrous events.

        Here is what you need to evaluate:

         ### Input:
         - **Keyword**: {examples['keyword']}
         - **Location**: {examples['location']}
         - **Text**: {examples['text']}

         Based on the provided information, determine whether the Tweet is about a real disaster. Answer only one word, which is 'Yes' or 'No'.
         """

        message = [
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        # print(input_text)
        if examples["target"]== 1:
            label =(f"Yes" + tokenizer.eos_token)
        else:
            label =(f"No" + tokenizer.eos_token)
        # print(label)
        # print(examples["target"])
        # print(type(examples["target"]))
        instruction = tokenizer(input_text,add_special_tokens=False)
        response = tokenizer(label,add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"]
        # print(len(input_ids))
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100]*len(instruction["input_ids"]) + response["input_ids"]
        # print(len(input_ids))
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        input_ids += [tokenizer.pad_token_id] * (max_length-len(input_ids))
        attention_mask += [0] * (max_length - len(attention_mask))
        labels += [-100] * (max_length-len(labels))
        # print(max_length-len(labels))
        # print(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    dataset = dataset.map(text_preprocess_function,remove_columns=dataset.column_names)
    print(dataset)
    return dataset


if __name__ == '__main__':
    train_dataset = get_dataset(train_path)
    train_dataset.shuffle()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map='auto',
        torch_dtype = torch.bfloat16
    )
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
       target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" ],
       r=16,
       lora_alpha=32,
       lora_dropout=0,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        bf16=True,
        output_dir=save_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=1000,
        report_to=[],
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=False)
    )
    model.config.use_cache = False
    trainer.train()
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)