#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer
import os
import pandas as pd

class MistralTrainer:
    def __init__(self, dataset_path, model_name_or_path, r, alpha, lora_dropout=0.1, bias="None",
                 task_type="CASUAL_LM", target_modules=["q_proj", "v_proj"],
                 output_dir="mistral-finetuned", per_device_train_batch_size=8,
                 gradient_accumulation_steps=1, optim="paged_adamw_32bit",
                 learning_rate=2e-4, lr_scheduler_type="cosine",
                 save_strategy="epoch", logging_steps=100, num_train_epochs=1,
                 max_steps=250, fp16=True, push_to_hub=False):
        self.dataset_path = dataset_path
        self.model_name_or_path = model_name_or_path
        self.r = r
        self.alpha = alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.save_strategy = save_strategy
        self.logging_steps = logging_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.fp16 = fp16
        self.push_to_hub = push_to_hub

    def prep_dataset(self):
        data_df = pd.read_csv(self.dataset_path)
        data_df["text"] = data_df[["ocr_text", "response"]].apply(
            lambda x: "###Human: Summarize this following dialogue: " + x["ocr_text"] + "\n###Assistant: " + x[
                "response"], axis=1)
        formatted_data = Dataset.from_pandas(data_df)
        return formatted_data

    def prep_model(self):
        formatted_data = self.prep_dataset()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config_loading,
            device_map="auto"
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=self.r, lora_alpha=self.alpha, lora_dropout=self.lora_dropout, bias=self.bias, task_type=self.task_type,
            target_modules=self.target_modules
        )

        model = get_peft_model(model, peft_config)

        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            save_strategy=self.save_strategy,
            logging_steps=self.logging_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            fp16=self.fp16,
            push_to_hub=False
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_data,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=1024
        )

        trainer.train()

def main():
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--model_name_or_path', type=str, help='Pretrained model name or path')
    parser.add_argument('--r', type=float, help='Value for r in LoraConfig')
    parser.add_argument('--alpha', type=float, help='Value for alpha in LoraConfig')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Value for lora_dropout in LoraConfig (default: 0.1)')
    parser.add_argument('--bias', type=str, default='None', help='Value for bias in LoraConfig (default: "None")')
    parser.add_argument('--task_type', type=str, default='CASUAL_LM', help='Value for task_type in LoraConfig (default: "CASUAL_LM")')
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "v_proj"], help='List of target modules in LoraConfig (default: ["q_proj", "v_proj"])')
    parser.add_argument('--output_dir', type=str, default='mistral-finetuned', help='Output directory for training results (default: "mistral-finetuned")')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size per device for training (default: 8)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation (default: 1)')
    parser.add_argument('--optim', type=str, default='paged_adamw_32bit', help='Optimizer choice (default: "paged_adamw_32bit")')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for training (default: 2e-4)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Type of learning rate scheduler (default: "cosine")')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy during training (default: "epoch")')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging frequency during training (default: 100)')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('--max_steps', type=int, default=250, help='Maximum number of training steps (default: 250)')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training (default: False)')
    parser.add_argument('--push_to_hub', action='store_true', help='Push results to the Hugging Face Model Hub (default: False)')


    

    args = parser.parse_args()

    trainer = MistralTrainer(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
        r=args.r,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,  # Add the remaining arguments here
        bias=args.bias,
        task_type=args.task_type,
        target_modules=args.target_modules,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub
    )

    trainer.prep_model()

if __name__ == "__main__":
    main()

