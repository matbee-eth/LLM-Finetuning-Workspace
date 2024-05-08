import torch
import safetensors
import random
import sys
import os
import random
from transformers import HfArgumentParser, TrainingArguments, AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, disable_caching
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader

DEVICE = "cuda:0"
USE_4_BIT = False
RESUME_FROM_CHECKPOINT = True

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            if image is None:
                continue
            question = example["query"]["en"]
            answer = random.choice(example["answers"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

def main(model_args, data_args, training_args, data_collator):
    print("model_args", model_args)
    train_dataset = load_dataset("parquet", data_files=[data_args.train_dataset], split="train") if data_args.train_dataset else load_dataset(data_args.dataset_name, split="train")
    eval_dataset = load_dataset("parquet", data_files=[data_args.eval_dataset], split="train") if data_args.eval_dataset else load_dataset(data_args.dataset_name, split="eval")
    default_training_args = {
        "num_train_epochs": 3,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "warmup_steps": 100,
        "learning_rate": 5e-5,
        "weight_decay": 0.1,
        "logging_steps": 10,
        "output_dir": "./docvqa_ft_tutorial",
        "save_strategy": "steps",
        "save_steps": 500,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "save_total_limit": None,
        "bf16": True,
        "remove_unused_columns": False,
        "save_safetensors": False,
        "neftune_noise_alpha": 5.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_checkpointing_kwargs": {"use_reentrant": True},
        "hub_strategy": "all_checkpoints",
        # "hub_model_id": "matbee/idefics2b-weblinx"
    }

    for key, value in default_training_args.items():
        setattr(training_args, key, value)
    print("training_args", training_args)
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=True,
    )
    if USE_4_BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype = getattr(torch, "bfloat16"),
            quantization_config=bnb_config,
            _attn_implementation="flash_attention_2",
            # use_cache=False,
        )
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=None,
            low_cpu_mem_usage=False,
        )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        bias="none",
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules='all-linear',
        task_type="CAUSAL_LM",
        use_dora=False
    )
    model = get_peft_model(model, lora_config)

    # model = PeftModel.from_pretrained(model, "/home/acidhax/dev/originals/LLM-Finetuning/weblinx/docvqa_ft_tutorial/checkpoint-20500") #get_peft_model(model, lora_config)
    # model = model.merge_and_unload()
    # model.save_pretrained("idefics2-weblinx-test")
    # model.push_to_hub("matbee/idefics2-weblinx-20500")

    # disable_caching()
    # train_dataset = train_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])
    # eval_dataset = load_dataset("nielsr/docvqa_1200_examples_donut", split="test") # TO CHANGE with nielsr/docvqa_1200_examples_donut
    # eval_dataset = eval_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])
    ##

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # hub_model_id="matbee/idefics2-weblinx"
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    # trainer.push_to_hub()