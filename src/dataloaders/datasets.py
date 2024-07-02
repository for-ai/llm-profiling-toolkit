import os
import json
import torch
import pandas as pd
from typing import List, Any
from torch.utils.data import Dataset
from datasets import Value
from datasets import Dataset as HFDataset
import datasets 

def load_dataset(
    dataset_name: str, 
    tokenizer: Any = None,
    **kwargs: dict,
    ) -> List[str]:
    """
    Function used to load data from a given dataset.

    Args:
        - dataset_name (str): name of the dataset to be loaded
        - dataset_path (str): absolute path of the dataset to be loaded
        - max_instances (int): number of instances to load from the dataset

    Returns:
        - List[str]: list of strings which comprise the loaded data
    """

    if dataset_name == 'StrategyQA':
        dataset = StrategyQADataset(tokenizer, max_instances=kwargs['max_instances'])
    elif dataset_name == 'Dolly200_val':
        dataset = Dolly200Dataset(tokenizer, max_instances=kwargs['max_instances'], test=False)
    elif dataset_name == 'Dolly200_test':
        dataset = Dolly200Dataset(tokenizer, max_instances=kwargs['max_instances'], test=True)
    elif dataset_name == 'OpenBookQA':
        dataset = OpenBookQADataset(tokenizer)
    elif dataset_name == 'HellaSwag':
        dataset = HellaSwagDataset(tokenizer)
    else:
        raise NotImplementedError("Dataset is invalid.")
    return dataset

class BaseDataset(Dataset):
    """
    Base instance of the dataset.
    """
    def __init__(self, tokenizer: Any = None):
        self.persistent_dir = os.path.realpath('.')
        self.tokenizer = tokenizer
    
    def __len__():
        raise NotImplementedError('Method has to be implemented before being used.')
    
    def __getitem__(self):
        raise NotImplementedError('Method has to be implemented before being used.')
    
    def _is_in_range(self, idx):
        if idx > abs(self.__len__()):
            raise ValueError(f'The index provided ({idx}) is out of range (should be between -{self.__len__()} and {self.__len__()}).')

    
class StrategyQADataset(BaseDataset):

    def __init__(self, tokenizer: Any = None, max_instances: int = None):
        super().__init__(tokenizer)
        self.data = datasets.load_dataset("wics/strategy-qa")['test'][:max_instances]['question']
        self.max_input_length = 128
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        input = self.data[idx]
        return input

class RTPDataset(BaseDataset):

    def __init__(self, tokenizer: Any = None, max_instances: int = None):
        super().__init__(tokenizer)
        self.data = datasets.load_dataset('json', data_files=os.path.join(self.persistent_dir, 'data/RTP/rtp_toxic_test_set.json'), split='train')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        input = self.data[idx]['prompt']
        return input

class Dolly200Dataset(BaseDataset):

    def __init__(self, tokenizer: Any = None, max_instances: int = None, test: bool = True):
        super().__init__(tokenizer)
        self.data = pd.read_json(os.path.join(self.persistent_dir, 'data/Dolly-200/dolly_eval_eng.jsonl'))
        self.data = self.data.loc[self.data['language'] =='eng']['inputs']
        self.data = self.data.iloc[100:] if test else self.data.iloc[:100]
        self.max_input_length = 512
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        input = self.data.iloc[idx]
        return input

class OpenBookQADataset(BaseDataset):

    def __init__(self, tokenizer: Any = None):
        super().__init__(tokenizer)
        self.data = datasets.load_dataset("openbookqa")['test']
        self.max_input_length = 256
        self.answer_dict = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        return self.data[idx]['question_stem'], self.data[idx]['choices']['text'], self.answer_dict[self.data[idx]['answerKey']]

class HellaSwagDataset(BaseDataset):

    def __init__(self, tokenizer: Any = None):
        super().__init__(tokenizer)
        self.data = datasets.load_dataset("Rowan/hellaswag")['validation']
        self.max_input_length = 1024
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        return self.data[idx]['ctx'], self.data[idx]['endings'], int(self.data[idx]['label'])

class BBQDataset(BaseDataset):

    def __init__(self, dataset_path: str, tokenizer: Any = None, lambda_filter: Any = None):
        super().__init__(tokenizer)
        self.data = datasets.load_dataset('json', data_files=dataset_path, split='train')
        if lambda_filter:
            self.data = self.data.filter(lambda_filter)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._is_in_range(idx)
        return  self.data[idx]['context'], \
                self.data[idx]['question'], \
                (self.data[idx]['ans0'], self.data[idx]['ans1'], self.data[idx]['ans2']), \
                self.data[idx]['answer_info'], \
                int(self.data[idx]['label']), \
                [g.lower() for g in self.data[idx]['additional_metadata']['stereotyped_groups']]

class DistillationDataset(BaseDataset):

    def __init__(self, dataset_path: str, 
                 prompt_func: Any = None, 
                 split: str = None, 
                 instruct_field: str = 'prompt',
                 local: bool = True,
                 dataset_format_test: bool = True):
        super().__init__(None)
        if local:
            self.data = datasets.load_dataset("json", data_files=dataset_path, split=split)
        else:
            self.data = datasets.load_dataset(dataset_path, split=split)
        self.prompt_func = prompt_func
        self.instruct_field = instruct_field
        if dataset_format_test:
            check_dataset_format(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            self._is_in_range(idx)
            formated_prompt = self.prompt_func(self.data[idx][self.instruct_field])
            return formated_prompt, self.data[idx][self.instruct_field]

def check_dataset_format(dataset):
    """
    Function to check whether dataset follows the correct format before passing it to trainer.

    Args:
        - dataset (Dataset): Dataset object with loaded dataset
    
    Returns:
        - None
    """
    assert type(dataset) == HFDataset, f"Dataset must be of type dataset.Dataset but is {type(dataset)}"
    assert {'completion', 'prompt'}.issubset(set(list(dataset.features.keys()))), f"Features present in dataset must be named ['completion', 'prompt'], but are {list(dataset.features.keys())}"
