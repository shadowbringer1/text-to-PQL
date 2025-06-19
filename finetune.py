import os
import json
import random
import sys
from typing import Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

scene_name_map = {
    "software_PSI": "软件PSI",
    "software_MPC": "软件MPC",
    "software_PIR": "软件PIR",
    "hardware_PSI": "硬件PSI",
    "hardware_MPC": "硬件MPC",
    "hardware_PIR": "硬件PIR",
    "hardware_PIRMPC": "硬件PIRMPC",
    "Federated_learning": "联邦学习"
}

def build_prompt(scene_name, chinese_question):
    scene_desc = scene_name_map.get(scene_name, "未知场景")
    return f"""你是一个专注于{scene_desc}任务的PQL生成助手，只需输出一条合法的PQL语句，不要添加任何解释或注释。请根据下方中文问题直接生成对应的PQL语句。

问题：{chinese_question}
PQL："""

def load_and_split_data(path: str, split_ratio=0.8):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_samples, val_samples = [], []

    for scene, items in raw_data.items():
        total = items
        random.shuffle(total)
        split_idx = int(len(total) * split_ratio)
        for item in total[:split_idx]:
            train_samples.append({
                "input": build_prompt(scene, item["Chinese_question"]),
                "label": item["PQL_query"]
            })
        for item in total[split_idx:]:
            val_samples.append({
                "input": build_prompt(scene, item["Chinese_question"]),
                "label": item["PQL_query"]
            })

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)

def tokenize_function(example, tokenizer, max_length):
    full_text = example["input"] + example["label"]
    model_inputs = tokenizer(full_text, max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def main():
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, val_dataset = load_and_split_data(cfg["data_path"], cfg["split_ratio"])
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg["max_length"]), remove_columns=["input", "label"])
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer, cfg["max_length"]), remove_columns=["input", "label"])

    training_args = TrainingArguments(
        output_dir=cfg["save_path"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=True,
        logging_dir="./logs",
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    model.save_pretrained(os.path.join(cfg["save_path"], "final"))
    tokenizer.save_pretrained(os.path.join(cfg["save_path"], "final"))

if __name__ == "__main__":
    main()