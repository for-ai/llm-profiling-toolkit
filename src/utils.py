import os
import torch
import random
import json
import numpy as np
from typing import Any
from transformers import (
    BitsAndBytesConfig
)
import json
import os
from typing import Any

def set_seed(seed: int):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_qa_answer(question: str, options: list[str], tokenizer: Any, model: Any) :
    """
    Function as defined in https://medium.com/nlplanet/zero-shot-question-answering-with-large-language-models-in-python-9964c55c3b38.
    Used to get model's preferences in QnA settings.

    Args:
        - question (str): question to be asked
        - options (str): alternatives to the question asked
        - tokenizer (Any): tokenizer to be used to turn prompt into tokens
        - model (Any): model from which the preferences will be extracted
    
    Returns:
        - scores (list[int]): list of scores for each alternative
        - chosen_alternative_idx (int): index of the chosen/prefered alternative
    """
    scores = []
    for o in options:
        input = tokenizer(question+' '+o, return_tensors="pt").input_ids.to('cuda:0')
        o_input = tokenizer(o, return_tensors="pt").to('cuda:0')
        o_len = o_input.input_ids.size(1)
        target_ids = input.clone()
        target_ids[:, :-o_len] = -100
        with torch.no_grad():
            outputs = model(input, labels=target_ids)
            neg_log_likelihood = outputs[0] 
        scores.append((-1*neg_log_likelihood.cpu()))
    args = np.argsort(scores)
    chosen_alternative_idx = torch.argmax(torch.tensor(scores), dim=-1)
    return scores, chosen_alternative_idx

def save_to_json(results_dict: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results_dict, f)

def get_quantization_config(quantize: bool, quantization_type: str, precision: str):
    """
    Function used to get quantization config to be passed to model loader.

    Args:
        - quantize (bool): flag to determine whether quantization will be used or not
        - quantization_type (str): what quantization type to use if quantize == True (choices: 4_bit or 8_bit)
        - precision (str): precision to be used for models (choices: regular, fp16 or bf16)
    
    
    Returns:
        - None if quantize == False else returns a BitsAndBytesConfig config object
    """
    if quantize:
        if quantization_type == '4_bit':
            if precision == 'regular':
                compute_dtype = getattr(torch, "float32")
            elif precision == 'fp16':
                compute_dtype = getattr(torch, "float16")
            elif precision == 'bf16':
                compute_dtype = getattr(torch, "bfloat16")
            else:
                ValueError(f'Precision type {precision} is not supported, it should be either regular, fp16 or bf16.')
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_type == '8_bit':
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            ValueError(f'Quantization type {quantization_type} is not supported, it should be either 4_bit or 8_bit')

    return None