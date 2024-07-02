"""
Script to profile a given LLM. 
"""

import os
import yaml
import argparse

from src.models import models
from src.utils import set_seed, get_quantization_config
from src.profiling import profiling_tools

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Parameters to perform profiling of a given model.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(thisdir),
    help="Directory where all persistent data will be stored, default to the directory of the cloned repository.",
)

parser.add_argument(
    "--model_type",
    action="store",
    type=str,
    default=None,
    choices=[
        "HuggingFaceModel",
        "AyaHuggingFace",
        "CohereModels"
    ],
    required=True,
    help="Model type to evaluate on, AutoModelForCausalLM models should use HuggingFaceModel.",
)

parser.add_argument(
    "--basemodel_path",
    action="store",
    default=None,
    type=str,
    required=True,
    help="Path to folder where model checkpoint is stored, both local checkpoints and remote HF paths can be used.",
)

parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=None,
    required=True,
    help="Max batch size to use to collect generations for TextualCharacteristicsProfiling.",
)

parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=None,
    required=True,
    help="Max number of tokens to be generated per generation for TextualCharacteristicsProfiling.",
)

parser.add_argument(
    "--text_dataset",
    action="store",
    type=str,
    default=None,
    choices=[
        "StrategyQA",
        "Dolly200_val",
        "Dolly200_test"
    ],
    help="Dataset to be used to prompt models to calculate textual characteristics.",
)

parser.add_argument(
    "--profiling_tools",
    action="store",
    type=lambda s: [item for item in s.split(',')],
    default="TextualCharacteristicsProfiling,SocialBiasProfiling,CalibrationProfiling,ToxicityProfiling",
    help="List of types of profiling tools to run separated by a comma (,), valid options are \
        TextualCharacteristicsProfiling,SocialBiasProfiling,CalibrationProfiling,ToxicityProfiling.",
)

parser.add_argument(
    "--experiment_dir",
    action="store",
    type=str,
    default='',
    help="Directory where results should be stored, if no directory name is provided defaults to <persistent_dir>/results/profiling/.",
)

parser.add_argument(
    "--quantize",
    action="store_true",
    default=False,
    help="Flag determining whether model should be quantized or not.",
)

parser.add_argument(
    "--quantization_type",
    action="store",
    type=str,
    default=None,
    choices=[
        "4_bit",
        "8_bit"
    ],
    help="What type of quantization to use.",
)

parser.add_argument(
    "--precision",
    action="store",
    type=str,
    choices=[
        "bf16",
        "fp16",
        "regular"
    ],
    default=None,
    help="Whether to use mixed-precision when training or not.",
)

parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=42,
    help="Seed value for reproducibility.",
)

parser.add_argument(
    "--auth_token",
    action="store",
    type=str,
    default=None,
    help="Hugginface authorization token necessary to run restricted models (e.g. LLaMa models).",
)

parser.add_argument(
    "--perspective_key",
    action="store",
    type=str,
    default=None,
    help="Perspective API key to use to perform ToxicityProfiling.",
)

def check_args(args):
    
    assert (args.perspective_key != None) == ('ToxicityProfiling' in args.profiling_tools), "Perspective API key is required to run ToxicityProfiling, \
otherwise --perspective_key arg should not be passed."
    assert (args.text_dataset != None) == ('TextualCharacteristicsProfiling' in args.profiling_tools), "Evaluation dataset name (--text_dataset) is required to run TextualCharacteristicsProfiling, \
otherwise --text_dataset arg should not be passed."

def main():
    args = parser.parse_args()
    check_args(args)

    set_seed(args.seed)
    quantization_config = get_quantization_config(args.quantize, args.quantization_type, args.precision)
    model = getattr(models, args.model_type)(args.basemodel_path, 
                                             args.auth_token, 
                                             quantization_config=quantization_config)
    if "HuggingFace" in args.model_type:
        model.model.eval()

    results = {}
    for profiling_tool_type in args.profiling_tools:
        tool = getattr(profiling_tools, profiling_tool_type)(args.persistent_dir, args.experiment_dir)
        results[profiling_tool_type] = tool(
            model, 
            max_new_tokens=args.max_new_tokens, 
            batch_size=args.batch_size, model_path=args.basemodel_path, 
            auth_token=args.auth_token,
            model_name=args.basemodel_path.split('/')[-1],
            text_dataset=args.text_dataset,
            perspective_key=args.perspective_key
        )

    # Metadata storage
    storage_path = os.path.join(args.persistent_dir, 'results/profiling', args.experiment_dir)
    with open(os.path.join(storage_path, 'metadata_profiling.yaml'), 'w') as metadata:
        yaml.dump(vars(args), metadata)

if __name__ == "__main__":
    main()