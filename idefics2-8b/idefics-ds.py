import torch
import safetensors
import random
import sys
import os
from transformers import HfArgumentParser, TrainingArguments, AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, disable_caching
from dataclasses import dataclass, field
from typing import Optional

DEVICE = "cuda:0"
USE_4_BIT = True
RESUME_FROM_CHECKPOINT = False

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="vikhyatk/moondream2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    use_loftq: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."},
    )
    use_loftq_callback: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."
        },
    )
    moe_layer_name: Optional[str] = field(
        default=None,
        metadata={"help": "MOE layer name"},
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


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
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
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
        "vikhyatk/moondream2",
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
            "vikhyatk/moondream2",
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
            "vikhyatk/moondream2",
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
    import random

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