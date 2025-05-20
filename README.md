# 🌟 Low-Rank Clone (LRC)

**Official Codebase for the paper:**
*A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone*.

🚀 **Paper Link**: [https://arxiv.org/abs/2505.12781](https://arxiv.org/abs/2505.12781)
🚀 **Model checkpoints are available on Hugging Face!**
👉 [Check them out here](https://huggingface.co/collections/JitaiHao/low-rank-clone-lrc-6828389e96a93f1d4219dfaf) 🔗

---

## 📚 Table of Contents

1.  [Environment Setup](#1-environment-setup)
2.  [Usage Examples](#2-usage-examples)
    *   [Training LRC-1.5B](#training-lrc-15b)
    *   [Checkpoint Conversion](#checkpoint-conversion)
    *   [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
3.  [Data Preparation](#3-data-preparation)
4.  [Evaluation](#4-evaluation)
5.  [Contact & Acknowledgments](#5-contact--acknowledgments)
6.  [Citation](#6-citation)

---

## 1. Environment Setup

LRC training has **strict `transformers` version requirements**. Please ensure you install the correct versions as detailed below.

### 🟢 LRC Training Environment

```bash
conda create -n lrc python=3.10 -y
conda activate lrc

# Install PyTorch
pip install torch==2.3.0

# Install core libraries with strict versioning for training
pip install transformers[torch]==4.41.2 \
            deepspeed==0.15.4 \
            accelerate==1.1.1 \
            datasets==2.19.2 \
            datatrove \
            fire \
            matplotlib \
            seaborn \
            wandb

# Install Flash Attention for optimized performance
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

### 🟡 `lm_eval` Environment

<details>
<summary>Why a separate environment for lm_eval?</summary>
The `lm_eval` library often requires a newer `transformers` version than what is compatible with the `deepspeed` setup used for LRC training. To avoid dependency conflicts, we recommend using a separate environment for evaluation.
</details>

```bash
conda create -n lm_eval --clone lrc
conda activate lm_eval

# Upgrade transformers for lm_eval compatibility
pip install transformers==4.51.3
pip install lm_eval==0.4.8
```

### 🟠 LlamaFactory Environment

```bash
conda create -n lf --clone lm_eval
conda activate lf

# Navigate to your LlamaFactory directory
cd /path/to/your/LlamaFactory

# Install LlamaFactory by following their official installation instructions
# (e.g., pip install -e .)
```

---

## 2. Usage Examples

### 🏋️ Training LRC-1.5B

Train the LRC-1.5B model using the following `accelerate` command:

```bash
accelerate launch --main_process_port 12231 --config_file "configs/accel_ds_8h800_gas1.yaml" hf_trainer.py \
  --log_steps 100 \
  --max_grad_norm 1.0 \
  --learning-rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --max_steps 208000 \
  --dataset_name ../datasets/mix_general_llama3_tokenized_v5.1/train.jsonl \
  --batch-size 3 \
  --data-max-len 2048 \
  --save_steps 20000 \
  --check_data_cls_loss False \
  --target_hidden_size 1536 \
  --kl_temperature 40 \
  --warmup-ratio 0.005 \
  --raw-model-name /path/to/your/TEACHER_MODEL \
  --extra_tags general_train,8h800,arch,try_sota,all_ffn,all_attn \
  --use_accelerate True \
  --output_dir ../ckpts \
  --str_ban_losses no \
  --tie_word_emb_proj 1 \
  --use_all_attn 1 \
  --aux_loss_scale_factor 0.2
```

#### 🔍 Key Arguments Explained

| Argument                       | Description                                                                                           |
| :----------------------------- | :---------------------------------------------------------------------------------------------------- |
| `accelerate launch`            | Command for distributed training via Hugging Face Accelerate.                                         |
| `--main_process_port`          | Port for the main process in distributed training.                                                    |
| `--config_file`                | Path to the Accelerate configuration file (e.g., `configs/accel_ds_8h800_gas1.yaml`).                 |
| `hf_trainer.py`                | The main training script leveraging Hugging Face Trainer.                                             |
| `--log_steps`                  | Frequency (in steps) to log training metrics.                                                         |
| `--max_grad_norm`              | Gradient clipping threshold to prevent exploding gradients.                                           |
| `--learning-rate`              | Initial learning rate for the optimizer.                                                              |
| `--gradient_accumulation_steps`| Number of steps to accumulate gradients before performing an optimizer step. Useful for larger effective batch sizes. |
| `--max_steps`                  | Total number of training steps.                                                                       |
| `--dataset_name`               | Path to your tokenized dataset in `jsonl` format.                                                     |
| `--batch-size`                 | Per-device batch size.                                                                                |
| `--data-max-len`               | Maximum input sequence length for training examples.                                                  |
| `--save_steps`                 | Frequency (in steps) to save model checkpoints.                                                      |
| `--check_data_cls_loss`        | Boolean flag (`True`/`False`) to enable/disable a specific auxiliary classification loss.             |
| `--target_hidden_size`         | The target hidden size for the student model (LRC). **Must match conversion and SFT.**                 |
| `--kl_temperature`             | Temperature parameter for KL divergence loss in knowledge distillation.                               |
| `--warmup-ratio`               | Ratio of total steps for learning rate warmup.                                                        |
| `--raw-model-name`             | Path to the teacher model (e.g., Llama-3.2-3B-Instruct). Used for configuration and initial weights. |
| `--extra_tags`                 | Comma-separated tags for experiment tracking (e.g., `wandb`).                                        |
| `--use_accelerate`             | Explicitly enable Accelerate features.                                                                |
| `--output_dir`                 | Directory to save checkpoints and training logs.                                                      |
| `--str_ban_losses`             | Comma-separated list of auxiliary losses to ignore (e.g., `mlp-gate-loss,attn-q-sim-loss`). Use `no` to enable all. |
| `--tie_word_emb_proj`          | Boolean flag (`1`/`0`) to tie word embeddings with the output projection layer (LM head).             |
| `--use_all_attn`               | Boolean flag (`1`/`0`) indicating if all attention layers are used in the student model.               |
| `--aux_loss_scale_factor`      | Scaling factor for the auxiliary (clone) loss.                                                       |

---

### ♻️ Checkpoint Conversion

After training, convert your LRC checkpoint into a standard Hugging Face-compatible student model.

```bash
python convert_ckpt.py \
  --ckpt-path /path/to/your/LRC_CKPT.safetensors \
  --target-hidden-size 1536 \
  --raw-model-name /path/to/your/TEACHER_MODEL \
  --save-path /path/to/save/your/STUDENT_MODEL \
  --use-all-attn 1 \
  --use-in-out-mlp 1 \
  --tie-word-emb-proj 1
```

**❗ Important:** The values for `--target-hidden-size`, `--use-all-attn`, `--use-in-out-mlp`, and `--tie-word-emb-proj` **MUST EXACTLY MATCH** those used during your LRC training!

#### Parameter Guide

| Argument              | Description                                                             | Example                                 |
| :-------------------- | :---------------------------------------------------------------------- | :-------------------------------------- |
| `--ckpt-path`         | Path to your LRC checkpoint file (e.g., `model.safetensors`).          | `../ckpts/lrc_model/model.safetensors`  |
| `--target-hidden-size`| Student model's hidden size. **Must match training config.**            | `1536`                                  |
| `--raw-model-name`    | Path to the teacher model (used for base configuration).                | `../models/Llama-3.2-3B-Instruct/`      |
| `--save-path`         | Directory where the converted student model will be saved.              | `../converted_models/student_model/`    |
| `--use-all-attn`      | Set to `1` if "all attention" was enabled during training, else `0`.    | `1` or `0`                              |
| `--use-in-out-mlp`    | Set to `1` if "in/out MLP" (FFN projection) was enabled, else `0`.      | `1` or `0`                              |
| `--tie-word-emb-proj` | Set to `1` if word embeddings were tied with the output projection, else `0`.| `1` or `0`                              |

---

### 🧑‍🎓 Supervised Fine-Tuning (SFT)

After converting your LRC checkpoint, you can fine-tune it using LlamaFactory.

```bash
# Ensure you are in the 'lf' conda environment (or equivalent LlamaFactory setup)
# and have navigated to your LlamaFactory directory.

# Make sure to update the 'model_name_or_path' in your YAML config!
FORCE_TORCHRUN=1 llamafactory-cli train /path/to/your/low-rank-clone/configs/llama_factory/llama3-sft-full.yaml
```

**Notes:**
*   `FORCE_TORCHRUN=1` is often required for `llamafactory-cli` to use `torchrun`.
*   You **must** modify the `model_name_or_path` field in the specified YAML configuration file (e.g., `llama3-sft-full.yaml`) to point to the directory of your newly converted LRC-trained model.

---

## 3. Data Preparation

All datasets are expected to be in pre-tokenized `jsonl` format.

### 🛠️ Example: Data Generation

```bash
# Ensure you are in the 'lrc' conda environment
python data/generate_general_data_parallel.py \
  --version v5.1 \
  --tkn-path /path/to/your/TEACHER_MODEL_TOKENIZER \
  --num-workers 8 \
  --data-max-len 2048
```

**Note:** The data paths within the `generate_general_data_parallel.py` script might be hardcoded. Please review and edit them as needed for your environment.

#### Key Arguments:

| Argument       | Description                                  |
| :------------- | :------------------------------------------- |
| `--version`    | Data generation version (e.g., `v5.1`).      |
| `--tkn-path`   | Path to the teacher model's tokenizer.       |
| `--num-workers`| Number of parallel workers for data processing.|
| `--data-max-len`| Maximum sequence length for each example.    |

---

## 4. Evaluation

**❗ Critical:** Run evaluation **ONLY within the `lm_eval` environment** to ensure correct `transformers` and `lm_eval` versions are used.

```bash
conda activate lm_eval

lm_eval \
  --model hf \
  --tasks "sciq,piqa,winogrande,arc_easy,logiqa,arc_challenge,boolq,mmlu,commonsense_qa" \
  --batch_size "auto" \
  --trust_remote_code \
  --num_fewshot 0 \
  --model_args pretrained=/path/to/your/CONVERTED_MODEL
```

#### Evaluation Arguments:

| Argument                        | Description                                                       |
| :------------------------------ | :---------------------------------------------------------------- |
| `--model hf`                    | Specifies the Hugging Face model interface for evaluation.        |
| `--tasks`                       | Comma-separated list of evaluation benchmarks from `lm_eval`.     |
| `--batch_size "auto"`           | Automatically selects the appropriate batch size.                 |
| `--trust_remote_code`           | Enables loading custom model hub code if required.                |
| `--num_fewshot 0`               | Number of examples for few-shot learning (0 for zero-shot).       |
| `--model_args pretrained=...`   | **Path to your trained and converted student model checkpoint.**  |

---

## 5. Contact & Acknowledgments

For any questions or issues, please open an issue on GitHub or contact us at [your_email@example.com] (if you wish to provide one).

We acknowledge and thank all the open-source projects and communities that made this work possible.

---

## 6. Citation

If you use our work, please cite our paper:

```bibtex
@misc{hao2025tokenworth1000tokens,
  title={A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone},
  author={Jitai Hao and Qiang Huang and Hao Liu and Xinyan Xiao and Zhaochun Ren and Jun Yu},
  year={2025},
  eprint={2505.12781},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.12781}
}
```