# LLM See, LLM Do: Guiding Data Generation to Target Non-Differentiable Objectives
> Luisa Shimabucoro, Sebastian Ruder, Julia Kreutzer, Marzieh Fadaee, Sara Hooker

[![arxiv](https://img.shields.io/badge/arXiv-2407.01490-b31b1b)](https://arxiv.org/abs/2407.01490) [![Static Badge](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/release/python-3110/)

Code for LLM profiling detailed in ["LLM See, LLM Do: Guiding Data Generation to Target Non-Differentiable Objectives"](https://arxiv.org/abs/2407.01490).

Currently we support base models from HuggingFace's [Transformers](https://github.com/huggingface/transformers) library in the PyTorch framework and Cohere models via the [Cohere API](https://docs.cohere.com/).


## 🖥️ Setup

Run the following to create the environment with all required dependencies:
```bash
conda create -n profiling python=3.11.7
conda activate profiling
pip install -r requirements.txt
pip install -e ~/profiling-toolkit/src/benchmarking/lm-evaluation-harness
pip install -e ~/profiling-toolkit/src/profiling/bias_bench
```

## 🧰 Usage

### 📊 Profiling LLMs
To profile a given LLM the following script should be run:
```Python
python run_profiling.py \
    --profiling_tools <profiling_tools> \
    --model_type <model_type> \
    --basemodel_path <basemodel_path> \ 
    --batch_size <batch_size> \ 
    --max_new_tokens <max_tokens> \ 
    --experiment_dir <experiment_dir> \ # optional
    --seed <seed> \ # optional
    --hf_auth_token <auth_token> \ # optional
    --quantize \ # optional
    --quantization_type <quant_type> \ # optional
    --precision <precision> \  # optional
    --text_dataset <text_dataset> \ # optional
    --perspective_key <perspective_key> \ # optional
```
<details>
<summary><code>>>> python run_profiling.py --help</code></summary>

```
Parameters to perform profiling of a given model.

options:
  -h, --help            show this help message and exit
  --persistent_dir PERSISTENT_DIR
                        Directory where all persistent data will be stored, default to the directory of the cloned repository.
  --model_type {HuggingFaceModel,AyaHuggingFace,CohereModels}
                        Model type to evaluate on, AutoModelForCausalLM models should use HuggingFaceModel.
  --basemodel_path BASEMODEL_PATH
                        Path to folder where model checkpoint is stored, both local checkpoints and remote HF paths can be used.
  --batch_size BATCH_SIZE
                        Max batch size to use to collect generations for TextualCharacteristicsProfiling.
  --max_new_tokens MAX_NEW_TOKENS
                        Max number of tokens to be generated per generation for TextualCharacteristicsProfiling.
  --text_dataset {StrategyQA,Dolly200_val,Dolly200_test}
                        Dataset to be used to prompt models to calculate textual characteristics.
  --profiling_tools PROFILING_TOOLS
                        List of types of profiling tools to run separated by a comma (,), valid options are TextualCharacteristicsProfiling,SocialBiasProfiling,CalibrationProfiling,ToxicityProfiling.
  --experiment_dir EXPERIMENT_DIR
                        Directory where results should be stored, if no directory name is provided defaults to <persistent_dir>/results/profiling/.
  --quantize            Flag determining whether model should be quantized or not.
  --quantization_type {4_bit,8_bit}
                        What type of quantization to use.
  --precision {bf16,fp16,regular}
                        Whether to use mixed-precision when training or not.
  --seed SEED           Seed value for reproducibility.
  --auth_token AUTH_TOKEN
                        Hugginface authorization token necessary to run restricted models (e.g. LLaMa models).
  --perspective_key PERSPECTIVE_KEY
                        Perspective API key to use to perform ToxicityProfiling.
```
</details>

### 🩺 Benchmark General Performance
To benchmark the general performance of a given LLM on a selection of tasks from lm-evaluation-harness the following script should be run:
```Python
python run_utility_benchmarking.py \
    --basemodel_path <basemodel_path> \ 
    --experiment_dir <experiment_dir> \ # optional
    --seed <seed> \ # optional
    --quantize \ # optional
    --quantization_type <quant_type> \ # optional
    --num_fewshot <num_fewshot> # optional
```
<details>
<summary><code>>>> python run_utility_benchmarking.py --help</code></summary>

```
Parameters to run general performance benchmarks.

options:
  -h, --help            show this help message and exit
  --persistent_dir PERSISTENT_DIR
                        Directory where all persistent data will be stored, default to the directory of the cloned repository.
  --basemodel_path BASEMODEL_PATH
                        Path to folder where model checkpoint is stored, both local checkpoints and remote HF paths can be used.
  --experiment_dir EXPERIMENT_DIR
                        Directory where results should be stored, if no directory name is provided defaults to <persistent_dir>/results/profiling/.
  --tasks TASKS         List of types of tasks from lm-evaluation-harness to evaluate your model on. To check the complete list of tasks run `lm-eval --tasks list`.
  --quantize            Flag determining whether model should be quantized or not.
  --quantization_type {4_bit,8_bit}
                        What type of quantization to use.
  --num_fewshot NUM_FEWSHOT
                        Number of few-shot examples to use during evaluation, default to 0.
  --seed SEED           Seed value for reproducibility.
```
</details>

## 📖 Metrics Overview

<table width="300">
    <thead>
        <tr>
            <th>Category</th>
            <th>Metric/Benchmark</th>
            <th>Overview</th>
            <th>Reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Toxicity</td>
            <td>Expected Maximum<br>Toxicity (EMT)</td>
            <td>Calculates the mean maximum toxicity scores by collecting k=25 generations for the same prompt. This is used to estimate worst-case toxicity. Scores are measured via the Perspective API.</td>
            <td rowspan=2><a href="https://arxiv.org/abs/2009.11462">RealToxicityPrompts</a> <a href="https://perspectiveapi.com/">Perspective API</a></td>
        </tr>
        <tr>
            <td>Toxicity Probability</td>
            <td>Calculates the empirical probability of a model generating at least one response with TOXICITY >= 0.5 over k=25 generations. This serves as a way to measure how frequently a model generates toxic responses.</td>
        </tr>
        <tr>
            <td rowspan=4>Social Bias</td>
            <td>SEAT</td>
            <td>The Sentence Encoder Association Test (SEAT) is an embedding-based benchmark that extends the Word Embedding Association Test (WEAT) to sentence-level representations. It evaluates bias by measuring the association strength between sets of attribute words (e.g., gender-related words) and sets of target words (e.g., family or career-related words).</td>
            <td><a href="https://aclanthology.org/N19-1063/">SEAT</a></td>
        </tr>
        <tr>
            <td>StereoSet</td>
            <td>StereoSet is a benchmark for measuring stereotypical bias in language models, using contexts with masked words and sets of stereotypical, anti-stereotypical, and unrelated associations. It quantifies bias by calculating a stereotype score, which is the percentage of examples where a model prefers stereotypical associations.</td>
            <td><a href="https://aclanthology.org/2021.acl-long.416/">StereoSet</a></td>
        </tr>
        <tr>
            <td>CrowS-Pairs</td>
            <td>Crowdsourced Stereotype Pairs (CrowS-Pairs) is a benchmark dataset that contains pairs of minimally distant sentences, with one sentence reflecting a stereotype and the other violating it. The benchmark quantifies bias in language models by measuring their preference for stereotypical sentences over anti-stereotypical ones, similarly to StereoSet but using a different set of comparison sentences.</td>
            <td><a href="https://aclanthology.org/2020.emnlp-main.154/">CrowS-Pairs</a></td>
        </tr>
        <tr>
            <td>BBQ</td>
            <td>BBQ (Bias in Question Answering) is a benchmark designed to measure social biases in the predictions of language models, particularly in question-answering tasks. It contains unique examples and templates, each consisting of two questions, answer choices, and two contexts: a partial context missing relevant information, and a disambiguating context that provides the necessary information.</td>
            <td><a href="https://aclanthology.org/2022.findings-acl.165/">BBQ</a></td>
        </tr>
        <tr>
            <td rowspan=5>Textual<br>Characteristics</td>
            <td>Measure of Textual<br>Lexical Diversity (MTLD)</td>
            <td>The Measure of Textual Lexical Diversity (MTLD) employs a sequential analysis of a body of text to estimate a lexical diversity score. MTLD reflects the average number of words in a row for which a certain TTR (Type Token Ratio) is maintained.</td>
            <td><a href="https://link.springer.com/article/10.3758/BRM.42.2.381">MTLD</a></td>
        </tr>
        <tr>
            <td>Length</td>
            <td>Calculates a group of metrics related to the length of generations: number of characters/tokens/sentences, sentence/token length etc</td>
            <td rowspan=4><a href="https://aclanthology.org/2022.findings-acl.165/">TextDescriptives</a></td>
        </tr>
        <tr>
            <td>Gunning-Fog</td>
            <td>Readability index that estimates the years of formal education needed to understand the text on a first reading. Grade level = 0.4 × (ASL + PHW) (ASL is the average sentence length (total words / total sentences), and PHW is the percentage of hard words (words with three or more syllables)).</td>
        </tr>
        <tr>
            <td>Rix</td>
            <td>Readability measure that estimates the difficulty of a text based on the proportion of long words (more than six characters) in the text. Rix = (n_long_words / n_sentences).</td>
        </tr>
        <tr>
            <td>Miscellaneous</td>
            <td>Aside from the metrics described above additional metrics and descriptive statistics are also computed and can be checked on the TextDescriptives reference.</td>
        </tr>
        <tr>
            <td rowspan=1>Calibration</td>
            <td>Expected Calibration Error</td>
            <td>The Expected Calibration Error (ECE) is a metric used to evaluate the reliability of a model's predicted probabilities. It does this by measuring the difference between accuracy and confidence across multiple bins of predictions. A lower ECE indicates better calibration, with a perfectly calibrated model achieving an ECE of zero. We calculate 1-bin and 10-bin ECE on HellaSwag and OpenBookQA.</td>
            <td rowspan=2><a href="https://arxiv.org/abs/2211.09110">HELM</a></td>
    </tbody>
</table>


## 🗣️ Acknowledgments

This repository makes use of code and/or data from the following repositories:

- [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models](https://github.com/McGill-NLP/bias-bench)
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Bias Benchmark for QA Dataset](https://github.com/nyu-mll/BBQ/tree/main)
- [On the Challenges of Using Black-Box APIs for Toxicity Evaluation in Research](https://github.com/for-ai/black-box-api-challenges)

We thank the authors for making their code publicly available.

## 📄 Citation

```latex
@misc{shimabucoro2024llmseellmdo,
      title={LLM See, LLM Do: Guiding Data Generation to Target Non-Differentiable Objectives}, 
      author={Luísa Shimabucoro and Sebastian Ruder and Julia Kreutzer and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2407.01490},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01490},}
```