import torch
import safetensors
import random
import sys
import os
import random
from transformers import HfArgumentParser, TrainingArguments, AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, disable_caching
from dataclasses import dataclass, field
from typing import Optional

DEVICE = "cuda:0"
USE_4_BIT = True
RESUME_FROM_CHECKPOINT = False


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
            print("TEXT", text)
            texts.append(text.strip())
            print("texts", texts)
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

def main(model_args, data_args, training_args):
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=True,
    )
    if USE_4_BIT:
        compute_dtype = getattr(torch, "bfloat16")
        quant_storage_stype = getattr(torch, "bfloat16")
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
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        # model.gradient_checkpointing_enable()
        # model = prepare_model_for_kbit_training(model)
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )#.to(DEVICE)

    ##

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
        use_dora=False
    )

    model = get_peft_model(model, lora_config)
    ##
    # disable_caching()
    train_dataset = load_dataset("nielsr/docvqa_1200_examples_donut", split="train") # TO CHANGE with nielsr/docvqa_1200_examples_donut
    # train_dataset = train_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])
    eval_dataset = load_dataset("nielsr/docvqa_1200_examples_donut", split="test") # TO CHANGE with nielsr/docvqa_1200_examples_donut
    # eval_dataset = eval_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])

    ##

    data_collator = MyDataCollator(processor)

    ##

    training_args = TrainingArguments(
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.1,
        logging_steps=10,
        output_dir="./docvqa_ft_tutorial",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed="zero_stage3_config.json",
        save_safetensors=False,
        neftune_noise_alpha=5.0,
        per_device_train_batch_size=1,
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    trainer.push_to_hub()

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)