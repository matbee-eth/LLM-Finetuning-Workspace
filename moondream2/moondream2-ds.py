import torch
import safetensors
import random
import sys
import os
import math
import inspect
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import HfArgumentParser, TrainingArguments, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, disable_caching
from dataclasses import dataclass, field
from typing import Optional
from deepspeed.accelerator import get_accelerator
from bitsandbytes.optim import Adam8bit
from tqdm import tqdm
from einops import rearrange
from PIL import Image 

DTYPE = torch.bfloat16
MD_REVISION = "2024-04-02"
USE_4_BIT = True
RESUME_FROM_CHECKPOINT = False

EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
LR = 1.5e-5

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
    trust_remote_code: Optional[bool] = field(
      default=False,
      metadata={"help": "enable remote code execution"}
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
    def __init__(self, processor, moondream):
        self.processor = processor
        self.moondream = moondream

    def __call__(self, batch):
        IMG_TOKENS = 729
        ANSWER_EOS = "<|endoftext|>"
        
        images = [sample['image'] for sample in batch]
        images = torch.stack(self.moondream.vision_encoder.preprocess(images))
        images = rearrange(images,
                          "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                          p1=14, p2=14)
        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [self.processor.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            image = sample['image']
            if image is None:
                continue
            question = sample["query"]["en"]
            answer = random.choice(sample["answers"])
            
            q_t = self.processor(
                f"\n\nQuestion: {question}\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))
            
            a_t = self.processor(
                f" {answer}{ANSWER_EOS}",
                add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)
            tokens_acc.append(toks)
            labels_acc.append(labs)
        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([self.processor.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        returning = (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )
        return returning
    
def compute_loss(batch, moondream, accelerator):
    images, tokens, labels, attn_mask = batch
    images = images.to(torch.cuda.current_device())
    tokens = tokens.to(torch.cuda.current_device())
    labels = labels.to(torch.cuda.current_device())
    attn_mask = attn_mask.to(torch.cuda.current_device())
    with torch.no_grad():
      img_embs = accelerator.unwrap_model(moondream).vision_encoder.encoder(images)
      img_embs = accelerator.unwrap_model(moondream).vision_encoder.projection(img_embs)
    tok_embs = accelerator.unwrap_model(moondream).text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = accelerator.unwrap_model(moondream).text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss

def main(model_args, data_args, training_args):
    accelerator = Accelerator()
    processor = AutoProcessor.from_pretrained(
        "vikhyatk/moondream2",
        do_image_splitting=True,
        trust_remote_code=True,
        revision=MD_REVISION
    )
    if USE_4_BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            # torch_dtype = getattr(torch, "bfloat16"),
            quantization_config=bnb_config,
            # low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            revision=MD_REVISION
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )#.to(DEVICE)

    ##

    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            'proj','fc1','fc2',
            'Wqkv','out_proj'
        ],
        task_type="CAUSAL_LM",
        use_dora=False
    )

    model = get_peft_model(model, lora_config)
    ##
    # disable_caching()
    def transforms(examples):
      examples["image"] = [image.convert("RGBA") for image in examples["image"]]
      return examples

    train_dataset = load_dataset("nielsr/docvqa_1200_examples_donut", split="train[:20]")
    # train_dataset.set_transform(transforms)
    # train_dataset = train_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])
    eval_dataset = load_dataset("nielsr/docvqa_1200_examples_donut", split="test[:20]") # TO CHANGE with nielsr/docvqa_1200_examples_donut
    # eval_dataset = eval_dataset.remove_columns(['questionId', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no'])
    data_collator = MyDataCollator(processor, model)


    ## For fine-tuning LoRA params
    lora_params = []
    for name, module in model.named_modules():
        if "lora" in name:
            lora_params.extend([p for p in module.parameters() if p.requires_grad])

    # To fine-tune all lora params (which can include the vision model)
    optimizer = Adam8bit(
        [
            {"params": lora_params},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )

    # # For fine-tuning all text model params
    # optimizer = Adam8bit(
    #     [
    #         {"params": moondream.text_model.parameters()},
    #     ],
    #     # [{"params": lora_params}],
    #     lr=LR * 0.1,
    #     betas=(0.9, 0.95),
    #     eps=1e-6
    # )

    # Cosine learning rate schedule.
    def lr_schedule(step, max_steps):
        x = step / max_steps
        if x < 0.1:
            return 0.1 * LR + 0.9 * LR * x / 0.1
        else:
            return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

    lora_alpha = 32
    lora_rank = 64
    if USE_4_BIT:
        LR_scaling = lora_alpha / (lora_rank**0.5)
    model, train_dataset, eval_dataset, test_dataloader, optimizer, lr_schedule = accelerator.prepare(
        model, train_dataset, eval_dataset, eval_dataset, optimizer, lr_schedule
    )
    
    dataloaders = {
      "train": DataLoader(
          train_dataset,
          batch_size=BATCH_SIZE,
          shuffle=True,
          collate_fn=data_collator,
      ),
      "test": DataLoader(
          eval_dataset,
          batch_size=3,
          collate_fn=data_collator,
      ),
    }
    total_steps = EPOCHS * len(train_dataset) // GRAD_ACCUM_STEPS
    model.module.text_model.train()
    model.module.text_model.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False},) #this fixes the no grad issues...
      
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    i = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            loss = compute_loss(batch, model, accelerator)
            accelerator.backward(loss)

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                if param_group['params'] == lora_params:
                    param_group['lr'] = lr * LR_scaling  # Apply scaling only to lora_params
                else:
                    param_group['lr'] = lr  # Apply base lr to all other params
    accelerator.wait_for_everyone()
    peft_model_id = f"moondream2_matbee"
    save_with_accelerate(accelerator, model, f"./output/{peft_model_id}")
    
    accelerator.wait_for_everyone()

def save_with_accelerate(accelerator, model, output_dir):
    print("SAVING MODEL", output_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_dir, 
        is_main_process=accelerator.is_main_process, 
        save_function=accelerator.save, 
        state_dict=state_dict,
        safe_serialization=False,
    )

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