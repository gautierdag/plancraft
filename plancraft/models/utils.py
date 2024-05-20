import glob
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_downloaded_models() -> dict:
    """
    Get the list of downloaded models on the NFS partition (EIDF).
    """
    downloaded_models = {}
    # known models on NFS partition
    if pathlib.Path("/nfs").exists():
        local_models = glob.glob("/nfs/public/hf/models/*/*")
        downloaded_models = {
            model.replace("/nfs/public/hf/models/", ""): model for model in local_models
        }
    return downloaded_models


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.longest_sequence_length = 0

    def insert(self, sequence: list):
        node = self.root
        for num in sequence:
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]
        node.is_end_of_sequence = True

        if len(sequence) > self.longest_sequence_length:
            self.longest_sequence_length = len(sequence)

    def starts_with(self, prefix: list) -> bool:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return False
            node = node.children[num]
        return True

    def get_next(self, prefix: list) -> list:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return []
            node = node.children[num]
        return list(node.children.keys())


def tokenize(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    max_tokens=256,
    new_message_start="act:",
):
    """
    Tokenize a list of messages and start the response message
    """
    message_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    message_text += new_message_start
    max_prompt_length = None
    # need to truncate if max_length is set
    if model.generation_config.max_length > max_tokens:
        max_prompt_length = model.generation_config.max_length - max_tokens

    tokenized_messages = tokenizer.encode(
        message_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    tokenized_messages = tokenized_messages.to(model.device)
    _, prompt_tokens = tokenized_messages.shape
    return tokenized_messages, prompt_tokens


def decode_with_choices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    choices: list[str],
    new_message_start: str = "act:",
    temperature=1.0,
) -> tuple[str, dict]:
    """
    Uses the Trie data structure to constrain the generation over a set of choices
    Returns the generated choice and the full generated sequence
    """
    tokenized_messages, prompt_tokens = tokenize(
        model, tokenizer, messages, new_message_start=new_message_start
    )

    class ValidActionsLogitsProcessor(torch.nn.Module):
        def __init__(self, choices: list[str]):
            super().__init__()
            self.choices = choices
            self.tree = Trie()
            self.step = 0
            self.start_idx = len(tokenized_messages[0])

            for choice in choices:
                tokenized_messages_with_choice, _ = tokenize(
                    model,
                    tokenizer,
                    messages,
                    new_message_start=f"{new_message_start}{choice}",
                )
                # find tokens that are different from the original message
                idxs = tokenized_messages_with_choice[0][self.start_idx :].tolist() + [
                    tokenizer.eos_token_id
                ]
                # insert the token idxs into the trie
                self.tree.insert(idxs)

        def forward(self, input_ids, scores):
            decoded_so_far = input_ids[0][self.start_idx :]
            valid_next_tokens = self.tree.get_next(decoded_so_far.tolist())
            mask = torch.full_like(scores, float("-inf"))
            mask[:, valid_next_tokens] = 0
            scores = scores + mask
            # print(scores[:, valid_next_tokens])
            return scores

    valid_actions = ValidActionsLogitsProcessor(choices)
    # Generate the initial action constrained to valid action tokens
    generated_sequence = model.generate(
        tokenized_messages,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=valid_actions.tree.longest_sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        use_cache=True,
        logits_processor=[valid_actions],
    )

    # select only new tokens
    generated_choice = generated_sequence[0][0][prompt_tokens:]
    # decode the generated choice
    generated_choice = tokenizer.decode(generated_choice, skip_special_tokens=True)
    return generated_choice, generated_sequence
