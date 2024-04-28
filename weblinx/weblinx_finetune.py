import datasets
import lxml.html
import json
from PIL import Image
from functools import partial
from typing import Callable
from pathlib import Path
from weblinx import Demonstration, filter_turns, Replay
from weblinx.processing.dom import clean_and_prune_tree
from weblinx.processing import load_candidate_elements
from weblinx.processing.prompt import (
    build_input_records_from_selected_turns,
    select_turns_and_candidates_for_prompts,
    find_turns_with_instructor_chat,
    format_candidates,
    format_utterances,
    format_utterances_truncated,
    get_speaker,
    multi_attempt_format_prev_turns_truncated,
)
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer
from weblinx.processing.truncation import (
    multi_attempt_truncate_cands_turn,
    multi_attempt_truncate_dom_tree,
)
import weblinx.utils.format as wlf
def build_formatter_for_multichoice():
    format_click = partial(wlf.format_click, formatters=(wlf.format_uid,))
    format_text_input = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text", max_length=200),
            wlf.format_uid,
        ),
    )
    format_change = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value", max_length=200),
            wlf.format_uid,
        ),
    )
    format_submit = partial(wlf.format_submit, formatters=(wlf.format_uid,))
    format_load = partial(
        wlf.format_load,
        include_transition=False,
        include_timestamp=False,
        max_length=200,
    )
    format_scroll = partial(wlf.format_scroll, include_timestamp=False)

    format_say = partial(wlf.format_say, include_timestamp=False)

    format_intent_auto = partial(
        wlf.format_intent_automatically,
        format_change=format_change,
        format_click=format_click,
        format_load=format_load,
        format_say=format_say,
        format_scroll=format_scroll,
        format_submit=format_submit,
        format_text_input=format_text_input,
    )

    return format_intent_auto


def get_system_prompt_template_for_llama_mc_concise(height=None, width=None):
    viewport_size = f"Viewport size: {height}h x {width}w ;\n" if height and width else ""
    sys_prompt_template = (
        "You are an AI assistant with a deep understanding of HTML "
        "and you must predict actions based on a user request, which will be executed. "
        "Use one of the following, replacing [] with an appropriate value: "
        "change(value=[str], uid=[str]) ; "
        "click(uid=[str]) ; "
        "load(url=[str]) ; "
        'say(speaker="navigator", utterance=[str]) ; '
        "scroll(x=[int], y=[int]) ; "
        "submit(uid=[str]) ;"
        "text_input(text=[str], uid=[str]) ;\n"
        "The user's first and last {num_utterances} utterances are: "
        "{utterance_context} ;\n" +
        viewport_size +
        "Only the last {num_prev_turns} turns are provided."
    )

    return sys_prompt_template


def get_candidate_prompt_template_for_llama():
    return "Here are the top candidates for this turn: {candidate_str}\n"


def get_final_user_message():
    return "Please select the best action using the correct format, do not provide any other information or explanation."


def merge_prev_turns(prev_turns_text_list, final_user_message):
    prev_turns_merged = []

    # Merge turns from the same role
    for i, turn_text in enumerate(prev_turns_text_list):
        role = get_speaker(
            turn_text,
            instructor_name="user",
            navigator_name="assistant",
            default_name="unknown",
        )

        if i > 0 and prev_turns_merged[-1]["role"] == role:
            prev_turns_merged[-1]["content"][-1]["text"] += " " + turn_text
        else:
            prev_turns_merged.append({"role": role, "content": [{"type": "text", "text": turn_text}]})

    if len(prev_turns_merged) > 0 and prev_turns_merged[-1]["role"] == "user":
        prev_turns_merged[-1]["content"] = [
            {
                "type": "text",
                "text": prev_turns_merged[-1]["content"][-1]["text"] + final_user_message
            },
            {
                "type": "image"
            }
        ]
    else:
        prev_turns_merged.append({"role": "user", "content": [
            {
                "type": "text",
                "text": final_user_message
            },
            {"type": "image"}
        ]})

    return prev_turns_merged


def build_prompt_records_for_llama_truncated(
    replay,
    turn,
    format_intent,
    tokenizer,
    processor,
    include_images=False,
    cands_turn=None,
    num_utterances=5,
    num_prev_turns=5,
    system_prompt_template=None,
    candidate_prompt_template=None,
    final_user_message=None,
    include_html=True,
    format_candidates_fn=partial(
        format_candidates, max_char_len=None, use_uid_as_rank=True
    ),
    merge_prev_turns_fn=merge_prev_turns,
    format_output_dict_fn: Callable = partial(
        wlf.format_output_dictionary, function_key="intent"
    ),
    max_html_tokens=700,
    max_utterance_tokens=40 * 5,
    max_prev_turns_tokens=50 * 5,
    max_candidates_tokens=65 * 10,
    add_unused_len_to_cands=True,
    allow_iterative_reduction=False,
    parser=None,
):
    """
    Parameters
    ----------
    ...
    allow_iterative_reduction : bool
        This arg is only relevant when truncate_at_center is used behind the scene (e.g. for
        multi_attempt_format_prev_turns_truncated or multi_attempt_truncate_dom_tree). If True,
        then we will allow the iterative reduction to continue until the max_tokens is reached.
        This is useful when the tokenizer output does not necessarily decrease when we remove
        tokens from the input. For example, if we remove a token that is part of a word, but
        the updated text is retokenized to the same number of tokens, then we will continue
        to remove tokens until we reach the max_tokens limit.
    """
    if system_prompt_template is None:
        system_prompt_template = get_system_prompt_template_for_llama_mc_concise()

    if candidate_prompt_template is None:
        candidate_prompt_template = get_candidate_prompt_template_for_llama()

    if final_user_message is None:
        final_user_message = get_final_user_message()

    instructor_chat_turns = find_turns_with_instructor_chat(
        replay, turn, num_prev_turns=num_prev_turns
    )
    utterance_context = format_utterances_truncated(
        instructor_chat_turns,
        tokenizer=tokenizer,
        max_tokens=max_utterance_tokens,
        num_utterances=num_utterances,
        format_utterances_fn=format_utterances,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_text_list = multi_attempt_format_prev_turns_truncated(
        replay=replay,
        turn=turn,
        format_intent=partial(format_intent, return_as=dict),
        tokenizer=tokenizer,
        num_prev_turns=num_prev_turns,
        turn_sep=None,  # output list
        max_tokens=max_prev_turns_tokens,
        max_attempts=5,
        format_output_dict_fn=format_output_dict_fn,
        warn_after_attempts=False,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_merged = merge_prev_turns_fn(
        prev_turns_text_list=prev_turns_text_list, final_user_message=final_user_message
    )

    sys_prompt = system_prompt_template.format(
        num_utterances=num_utterances - 1,  # 1 less since we add the first utterance
        utterance_context=utterance_context,
        height=turn.viewport_height,
        width=turn.viewport_width,
        num_prev_turns=num_prev_turns,
    )

    if include_html and turn.html not in ["", None] and cands_turn is not None:
        dom_tree_raw = lxml.html.fromstring(turn.html, parser=parser)
        dom_tree_pruned = clean_and_prune_tree(dom_tree_raw, cands_turn=cands_turn)
        trunc = multi_attempt_truncate_dom_tree(
            dom_tree=dom_tree_pruned,
            tokenizer=tokenizer,
            max_tokens=max_html_tokens,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        html = trunc["tree_repr"]
        sys_prompt = f"```html\n{html}\n```\n" + sys_prompt
    else:
        html = ""

    if cands_turn is not None:
        if add_unused_len_to_cands:
            # Add the unused length to the candidates
            num_html_tokens = len(tokenizer.tokenize(html))
            num_utter_tokens = len(tokenizer.tokenize(utterance_context))
            num_prev_turns_tokens = len(
                tokenizer.tokenize(" ".join(prev_turns_text_list))
            )
            remain_html_tokens = max_html_tokens - num_html_tokens
            remain_utter_tokens = max_utterance_tokens - num_utter_tokens
            remain_prev_turns_tokens = max_prev_turns_tokens - num_prev_turns_tokens
            remain_tokens = (
                remain_html_tokens + remain_utter_tokens + remain_prev_turns_tokens
            )
            # Add the unused length to the max_candidates_tokens
            max_candidates_tokens += remain_tokens

        cands_turn_trunc = multi_attempt_truncate_cands_turn(
            cands_turn=cands_turn,
            tokenizer=tokenizer,
            max_tokens=max_candidates_tokens,
            format_candidates_fn=format_candidates_fn,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        cand_str = format_candidates_fn(cands_turn_trunc, max_char_len=None)
        cand_prompt = candidate_prompt_template.format(candidate_str=cand_str)
        sys_prompt += "\n" + cand_prompt

    return [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, *prev_turns_merged]


def __insert_empty_user_content_at_first(prompt: list):
    """
    Given a list of dictionary representing the input prompt, insert an empty user content at the first position
    after system content, only if it is not already a user content. This is done in place.
    """
    if prompt[0]["role"] != "system":
        raise ValueError(
            f"First prompt must be a system prompt. Got {prompt[0]['role']} instead."
        )

    if prompt[1]["role"] != "user":
        prompt.insert(1, {"role": "user", "content": [{"type": "text", "text": ""}]})


def insert_formatted_chat_into_records(
    records,
    processor,
    include_output_target=True,
    origin_key="prompt",
    text_key="text",
):
    """
    Given a list of records, insert the formatted chat into the records. This is done in place.
    Note that we need a tokenizer's `apply_chat_template` method to be available.
    """
    for i, record in enumerate(records):
        __insert_empty_user_content_at_first(record[origin_key])

        if include_output_target:
            combined = record[origin_key] + [{"role": "assistant", "content": [{
                "type": "text",
                "text": record["output_target"]
            }]}]
            combined
        else:
            combined = record[origin_key]

        text = processor.apply_chat_template(
            combined, tokenize=False, add_generation_prompt=False
        )
        records[i][text_key] = text
from PIL import Image

def main(processor, tokenizer):
    wl_dir = Path("/media/acidhax/data/datasets/webLINX-full/")
    base_dir = wl_dir / "demonstrations"
    split_path = wl_dir / "splits.json"
    candidates_path = wl_dir / "candidates/train.jsonl"

    # Load the name of the demonstrations in the training split
    demo_names = ['aaabtsd', 'frvoxei', 'iuealcb', "apcwqgp", "bhcfixm", "cihwsqb", "dgcfhzx", "dwilaqa", "ejswgyk", "fcalwwv", "ftfmtwv", "gmlesoh", "hhzcqzv", "hzrykcf", "iwtxybo", "jptsssp", "kimiuov", "ldasoxo", "lyjvevw", "mrysrpr", "nrfopnt", "ooyygpw", "plvhjif", "qgtzogg", "rbkfraa", "rtjqoxa"]
    
    demos = [Demonstration(name, base_dir=base_dir) for name in demo_names]
    candidates = load_candidate_elements(candidates_path, group_keys=('demo_name', 'turn_index'), log_fn=None)

    format_intent = build_formatter_for_multichoice()

    selected_turns = select_turns_and_candidates_for_prompts(
        demos=demos,
        candidates=candidates,
        num_candidates=10,
    )
    
    # Build input records for training the model
    input_records = build_input_records_from_selected_turns(
        selected_turns=selected_turns,
        format_intent=format_intent,
        build_prompt_records_fn=partial(
            build_prompt_records_for_llama_truncated,
            format_intent=format_intent,
            tokenizer=processor.tokenizer,
            include_images=True,
            processor=processor,
        ),
        format_prompt_records_fn=None
    )

    insert_formatted_chat_into_records(
        input_records, processor, include_output_target=True
    )
    def loadImage(item):
        print("loadImage", item["image"])
        item["image"] = [Image.open(item["image"][0])]
        return item
    input_records_texts = [{"text": record["text"], "image": record["screenshot_path"]} for record in input_records]
    input_records_fname = "input_records_trunc.json"
    with open(input_records_fname, "w") as f:
        json.dump(input_records_texts, f, indent=2)
    dataset = datasets.Dataset.from_list(input_records_texts)
    dataset.set_transform(loadImage)
    
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
            text = example["text"]
            if image is None:
                continue
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

if __name__ == "__main__":
    MD_REVISION = None
    
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=True,
        trust_remote_code=True,
        revision=MD_REVISION
    )
    # print(processor.apply_chat_template(test, tokenize=False))
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=True,
        trust_remote_code=True,
        use_fast=True
     )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.chat_template = IDEFICS2_CHAT_TEMPLATE

    main(processor, tokenizer)