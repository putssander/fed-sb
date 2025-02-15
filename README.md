## Environment Setup

First, create and activate a Conda environment with Python 3.10, then install the necessary libraries by running:

```bash
conda create -n fed-sb python=3.10
conda activate fed-sb
pip install -r requirements.txt
pip install -e .
cd fed_sb
```

## Arithmetic Reasoning

To fine-tune a model on the MetaMathQA dataset and evaluate its performance on GSM8K and MATH benchmarks, execute the following script:

```bash
bash fed/scripts/arithmetic.sh
```

You can modify the `BASE_MODEL` parameter within the script to experiment with different models.

## Commonsense Reasoning

### Dataset Preparation

1. **Fine-tuning Data:**  
   Download the fine-tuning dataset from [this link](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json) and place it in the `data/commonsense` directory.

2. **Evaluation Data:**  
   Download the evaluation datasets from [this repository](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset). Save each dataset in its corresponding subdirectory under `data/commonsense`.

### Running the Experiments

Once the datasets are in place, run the experiments with:

```bash
bash fed/scripts/cr.sh
```

This script fine-tunes a model on the Commonsense170K dataset and evaluates it across eight different datasets. The `BASE_MODEL` parameter is configurable, allowing you to test various models.

## Privacy-Preserving Fine-Tuning

### Dataset Preparation

Download the SNLI dataset from [this link](https://nlp.stanford.edu/projects/snli/snli_1.0.zip), unzip it, and store the contents in the `DP/SNLI/data` directory.

### Running the Experiments

There are two scenarios for privacy-preserving fine-tuning:

- **Centralized Private Fine-Tuning:**  
  Run the following script:

  ```bash
  bash DP/SNLI/scripts/central_private.sh
  ```

- **Federated Private Fine-Tuning:**  
  Run the following script:

  ```bash
  bash DP/SNLI/scripts/fed_private.sh
  ```

In both cases, you can adjust the `epsilon` parameter as needed to tune the level of privacy.

---
