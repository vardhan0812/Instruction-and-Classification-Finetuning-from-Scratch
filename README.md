# LLM Fine-Tuning Notebook

This project contains a single Jupyter notebook, `LLM_Finetuning.ipynb`, that walks through two practical large language model fine-tuning tasks built around GPT-2:

1. SMS spam classification with a custom classification head
2. Instruction fine-tuning for response generation

The notebook starts from loading pretrained GPT-2 weights, builds model components in PyTorch, and then adapts the model for both classification and instruction-following workflows.

## File

- `LLM_Finetuning.ipynb`: main notebook containing the full end-to-end pipeline

## What The Notebook Covers

### 1. Loading pretrained GPT-2 weights

The notebook downloads official GPT-2 checkpoint files from OpenAI storage and loads them into a custom PyTorch implementation of the model.

Topics included:

- downloading GPT-2 model files
- building transformer components from scratch
- tokenization with `tiktoken`
- weight mapping from downloaded GPT-2 checkpoints into the PyTorch model

### 2. Classification fine-tuning

The first fine-tuning task turns GPT-2 into a spam classifier using the UCI SMS Spam Collection dataset.

This section includes:

- downloading and extracting the SMS Spam Collection dataset
- balancing spam vs. ham samples
- train/validation/test splitting
- adding a classification head
- freezing pretrained layers
- training and evaluation for spam detection
- running inference on example text messages

### 3. Instruction fine-tuning

The second fine-tuning task uses an instruction dataset stored in `instruction-data.json` and teaches the model to generate task-completing responses.

This section includes:

- downloading the instruction dataset JSON
- formatting prompts with instruction, optional input, and response blocks
- creating tokenized training samples
- masking target token IDs for training
- building custom dataloaders
- loading a pretrained GPT-2 variant for generation
- fine-tuning and evaluating generated responses
- exporting results to `instruction-data-with-response.json`

## Requirements

The notebook uses Python with the following main libraries:

- `torch`
- `tensorflow>=2.15.0`
- `numpy`
- `pandas`
- `tqdm`
- `requests`
- `tiktoken`

Depending on your environment, you may need to install some packages manually before running the notebook.

Example:

```bash
pip install torch tensorflow>=2.15.0 numpy pandas tqdm requests tiktoken
```

## Expected Downloads And Generated Files

While running, the notebook may create or download:

- `gpt2/` for GPT-2 checkpoints
- `sms_spam_collection.zip`
- `sms_spam_collection/SMSSpamCollection.tsv`
- `instruction-data.json`
- `instruction-data-with-response.json`

## How To Run

1. Open `LLM_Finetuning.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells from top to bottom.
3. Make sure package installs complete before executing later sections.
4. If using a GPU-enabled environment, the notebook will use CUDA when available.

## Notes

- This notebook is organized as a step-by-step implementation covering both model setup and fine-tuning workflows.
- Both workflows are kept in the same notebook, so running everything end to end may take time and download multiple assets.
- A GPU-enabled environment is recommended for the fine-tuning sections.
