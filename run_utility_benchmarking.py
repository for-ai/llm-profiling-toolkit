"""
Script to benchmark pretrained model. 
"""
from lm_eval import tasks, evaluator
import fnmatch
import os
import yaml
import json
import argparse

from typing import Any
from src.utils import set_seed

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Parameters to run general performance benchmarks.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(thisdir),
    help="Directory where all persistent data will be stored, default to the directory of the cloned repository.",
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
    "--experiment_dir",
    action="store",
    type=str,
    default='',
    help="Directory where results should be stored, if no directory name is provided defaults to <persistent_dir>/results/profiling/.",
)

parser.add_argument(
    "--tasks",
    action="store",
    type=lambda s: [item for item in s.split(',')],
    default="",
    help="List of types of tasks from lm-evaluation-harness to evaluate your model on. To check the complete list of tasks run `lm-eval --tasks list`.",
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
    "--num_fewshot",
    action="store",
    type=int,
    default=0,
    help="Number of few-shot examples to use during evaluation, default to 0.",
)

parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=42,
    help="Seed value for reproducibility.",
)

def eval_few_shot(basemodel_path: str,
                   task_list: list = ["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
                   quantization_config: Any = None,
                   num_fewshot: int = 0, 
                   seed : int = None) -> dict: 
    """
    Function to benchmark model on on the tasks listed.

    Args:
        - model_path (str): path where model checkpoints are stored so it can be loaded
        - task_list (str): list of tasks/benchmarks to evaluate the model on (to check which benchmarks are supported run `lm_eval --tasks list`)
        - num_shot (int): shot number to consider when running benchmarks (if num_fewshot == 0 then benchmarks are run on a zero-shot setting), defaults to zero shot
        - seed (int): seed to be set to account for reproducibility
    
    Returns:
        - results (dict): results for each benchmark to be saved in a .json file 
    """
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_manager = tasks.TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    model_args = {'pretrained' : basemodel_path}
    if quantization_config['quantize']:
        model_args['load_in_4bit'] = quantization_config['quantization_type'] == '4_bit'
        model_args['load_in_8bit'] = quantization_config['quantization_type'] == '8_bit'
        
    limit = None 
    if "70b" in basemodel_path or "65b" in basemodel_path:
        limit = 2000
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        use_cache=None,
        limit=limit,
        check_integrity=False,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed
    )

    return results 

def main():
    args = parser.parse_args()

    set_seed(args.seed)
    quantization_config = {'quantize' : args.quantize, 'quantization_type' : args.quantization_type}
    results = eval_few_shot(args.basemodel_path,
                            task_list=args.tasks,
                            quantization_config=quantization_config, 
                            num_fewshot=args.num_fewshot, 
                            seed=args.seed)

    # Result storage
    storage_path = os.path.join(args.persistent_dir, 'results/profiling', args.experiment_dir)
    os.makedirs(storage_path, exist_ok=True)
    model_name = args.basemodel_path.split("/")[-1]
    with open(os.path.join(storage_path, f'utility_benchmarks_{model_name}.json'), "w") as f:
        json.dump(results, f)
    
    with open(os.path.join(storage_path, 'metadata_benchmarking.yaml'), 'w') as metadata:
        yaml.dump(vars(args), metadata)

if __name__ == "__main__":
    main()