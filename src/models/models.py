import transformers
from peft import PeftModel
import os
import numpy as np
import torch
import cohere

class HuggingFaceModel:
    def __init__(self, model_name_or_path, auth_token: str = None, **kwargs: dict):
        self.model =  transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                             device_map='auto',
                                                             token=auth_token,
                                                             output_hidden_states=True,
                                                             quantization_config=kwargs.get('quantization_config'),
                                                             attn_implementation=kwargs.get('attn_implementation')
                                                             ).bfloat16()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, token=auth_token, padding_side='left')
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.instruct = True
    
    def generate(self, prompts, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding='max_length', max_length=kwargs['max_input_length'])['input_ids']
        if num_completions > 1:
            if kwargs.get('toxicity'):
                generate_ids = self.model.generate(tokenized_prompts.cuda(), 
                    max_new_tokens=max_new_tokens, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    use_cache=True, 
                    repetition_penalty=kwargs['repetition_penalty'],
                    num_return_sequences=num_completions, 
                    top_p=0.90,
                    do_sample=True,
                    diversity_penalty=0.0, 
                    temperature=1.0)              
        else:
            generate_ids = self.model.generate(tokenized_prompts.cuda(), 
                                            max_new_tokens=max_new_tokens, 
                                            pad_token_id=self.tokenizer.pad_token_id, 
                                            use_cache=True, 
                                            repetition_penalty=kwargs['repetition_penalty'])
        decoded_texts = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded_texts
    
    def format_instruct(self, prompts):
        formated_prompts = []
        for prompt in prompts:
            formated_prompts.append(f"### Question: {prompt}\n ### Answer: ")
        
        return formated_prompts
    
    def format_completion(self, prompts):
        formated_prompts = []
        for prompt in prompts:
            formated_prompts.append(f"### Instruction: Generate a continuation to the following input sentence, DO NOT answer with anything else BUT the continuation. \n### Input: {prompt} \n### Continuation: ")
        
        return formated_prompts
    
    def get_generation(self, generations, generation_divider: str = '### Answer: '):
        answers = []
        for generation in generations:
            answers.append(generation.split(generation_divider, 1)[-1])
        
        return answers
    
    def generate_distillation(self, prompts, questions, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        instances = []
        if num_completions > 1:
            generations = self.generate(prompts, max_new_tokens, num_completions, **kwargs)
            raw_generations = self.get_generation(generations, kwargs['generation_divider'])
            repeated_questions = np.repeat(questions, num_completions)
            n_unique_prompts = int(len(raw_generations) / num_completions)
            for prompt_idx in range(n_unique_prompts):
                instance_dict = {'prompt' : repeated_questions[prompt_idx*num_completions]}
                for completion_idx in range(num_completions):
                    instance_dict[f'completion_{completion_idx}'] = raw_generations[(prompt_idx*num_completions)+completion_idx]
                instances.append(instance_dict)
        else:
            generations = self.generate(prompts, max_new_tokens, **kwargs)
            raw_generations = self.get_generation(generations, kwargs['generation_divider'])
            for question, answer in zip(questions, raw_generations):
                instances.append({'prompt' : question, 'completion' : answer})
        
        return instances

class AyaHuggingFace:
    def __init__(self, model_name_or_path, auth_token: str = None, **kwargs: dict):
        self.model =  transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                             device_map='auto',
                                                             token=auth_token,
                                                             output_hidden_states=True,
                                                             quantization_config=kwargs.get('quantization_config'),
                                                             attn_implementation=kwargs.get('attn_implementation')
                                                             ).bfloat16()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, token=auth_token, padding_side='left')
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.instruct = False
    
    def generate(self, prompts, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        chat_prompts = []
        for prompt in prompts:
            message = {"role": "user", "content": prompt}
            conversation = [self.format_completion(), message] if kwargs.get('completion') else [message]
            chat_prompts.append(self.tokenizer.apply_chat_template(conversation=conversation, tokenize=True, padding='max_length', max_length=kwargs['max_input_length'], add_generation_prompt=True, return_tensors="pt"))
        if num_completions > 1:
            if kwargs.get('toxicity'):
                generate_ids = self.model.generate(torch.cat(chat_prompts, 0).cuda(), 
                    max_new_tokens=max_new_tokens, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    use_cache=True, 
                    repetition_penalty=kwargs['repetition_penalty'],
                    num_return_sequences=num_completions, 
                    top_p=0.90,
                    do_sample=True,
                    diversity_penalty=0.0, 
                    temperature=1.0)  
            else:
                generate_ids = self.model.generate(torch.cat(chat_prompts, 0).cuda(), 
                                                max_new_tokens=max_new_tokens, 
                                                pad_token_id=self.tokenizer.pad_token_id, 
                                                use_cache=True, 
                                                repetition_penalty=kwargs['repetition_penalty'],
                                                num_return_sequences=num_completions, 
                                                num_beams=num_completions, 
                                                num_beam_groups=num_completions, 
                                                do_sample=False, 
                                                diversity_penalty=1.0, 
                                                temperature=1.0) 
        else:
            generate_ids = self.model.generate(torch.cat(chat_prompts, 0).cuda(), 
                                           max_new_tokens=max_new_tokens, 
                                           pad_token_id=self.tokenizer.pad_token_id, 
                                           use_cache=True, 
                                           repetition_penalty=kwargs['repetition_penalty'])
        decoded_texts = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded_texts

    def format_completion(self):
        system_message = {'role' : 'system', 'content' : "Generate a continuation to the following input sentence, DO NOT answer with anything else BUT the continuation."}
        return system_message

    def get_generation(self, generations, **kwargs):
        answers = []
        for generation in generations:
            answers.append(generation.split('<|CHATBOT_TOKEN|>', 1)[-1])
        
        return answers

    def generate_distillation(self, prompts, questions, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        instances = []
        if num_completions > 1:
            generations = self.generate(questions, max_new_tokens, num_completions, **kwargs)
            raw_generations = self.get_generation(generations)
            repeated_questions = np.repeat(questions, num_completions)
            n_unique_prompts = int(len(raw_generations) / num_completions)
            for prompt_idx in range(n_unique_prompts):
                instance_dict = {'prompt' : repeated_questions[prompt_idx*num_completions]}
                for completion_idx in range(num_completions):
                    instance_dict[f'completion_{completion_idx}'] = raw_generations[(prompt_idx*num_completions)+completion_idx]
                instances.append(instance_dict)
        else:
            generations = self.generate(questions, max_new_tokens, num_completions, **kwargs)
            raw_generations = self.get_generation(generations)
            for question, answer in zip(questions, raw_generations):
                instances.append({'prompt' : question, 'completion' : answer})
        
        return instances

class CohereModels:
    def __init__(self, model_name_or_path, auth_token: str = None, **kwargs: dict):
        self.model = cohere.Client(auth_token)
        self.model_name = model_name_or_path
        self.tokenizer = None
        self.instruct = False
    
    def generate(self, prompts, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        generations = []
        for prompt in prompts:
            for _ in range(num_completions):
                response = self.model.chat(message=prompt, 
                                        model=self.model_name, 
                                        max_tokens=max_new_tokens, 
                                        temperature=1.,
                                        preamble=self.format_completion() if kwargs.get('completion') else None)
                generations.append(response.text)
        
        return generations

    def format_completion(self):
        preamble = "Generate a continuation to the following input sentence, DO NOT answer with anything else BUT the continuation." 
        return preamble

    def generate_distillation(self, prompts, questions, max_new_tokens: int = 512, num_completions: int = 1, **kwargs):
        instances = []
        if num_completions > 1:
            generations = self.generate(prompts, max_new_tokens, num_completions, **kwargs)
            repeated_questions = np.repeat(questions, num_completions)
            n_unique_prompts = int(len(generations) / num_completions)
            for prompt_idx in range(n_unique_prompts):
                instance_dict = {'prompt' : repeated_questions[prompt_idx*num_completions]}
                for completion_idx in range(num_completions):
                    instance_dict[f'completion_{completion_idx}'] = generations[(prompt_idx*num_completions)+completion_idx]
                instances.append(instance_dict)
        else:
            raw_generations = self.generate(questions, max_new_tokens, **kwargs)
            for question, answer in zip(questions, raw_generations):
                instances.append({'prompt' : question, 'completion' : answer})
        
        return instances
            