# 🌟 Low-Rank Clone (LRC)

**Code for the paper** *A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone*.

🚀 **Model checkpoints are available on Hugging Face!**  
👉 [Check them out here](https://huggingface.co/collections/JitaiHao/low-rank-clone-lrc-6828389e96a93f1d4219dfaf) 🔗


---

## 🚀 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Usage Examples](#usage-examples)
   - [Training LRC-1.5B](#training-lrc-15b)
   - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
   - [Checkpoint Conversion](#checkpoint-conversion)
3. [Data Preparation](#data-preparation)
4. [Evaluation](#evaluation)
5. [Contact & Acknowledgments](#contact--acknowledgments)

---

## 1. Environment Setup

> **Important:**  
> LRC training requires `transformers` version **≤ 4.41.2**. Please ensure you install the correct versions as detailed below.

### 🟢 LRC Training Environment

```bash
conda create -n lrc python=3.10 -y
conda activate lrc

pip install torch==2.3.0
pip install transformers[torch]==4.41.2
pip install deepspeed==0.15.4
pip install fire matplotlib seaborn datasets==2.19.2 datatrove
pip install wandb
pip install accelerate==1.1.1
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

---

### 🟡 lm_eval Environment

```bash
conda create -n lm_eval --clone lrc
conda activate lm_eval

pip install transformers==4.51.3
pip install lm_eval==0.4.8
```

---

### 🟠 LlamaFactory Environment

```bash
conda create -n lf --clone lm_eval
conda activate lf

# Navigate to your LlamaFactory directory
cd llamafactory_PATH 

# Install LlamaFactory (run the appropriate install scripts/commands as required)
```

## 2. Usage Examples


### 🏋️ Training LRC-1.5B

Train the LRC-1.5B model using the following command:

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
  --raw-model-name TEACHER_MODEL_PATH \
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
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `accelerate launch`            | Distributed training via Hugging Face Accelerate                                                      |
| `--main_process_port`          | Port for main process (distributed training)                                                          |
| `--config_file`                | Path to Accelerate config (e.g., `configs/accel_ds_8h800_gas1.yaml`)                                  |
| `hf_trainer.py`                | Training script (Hugging Face Trainer)                                                                |
| `--log_steps`                  | Log metrics every N steps                                                                             |
| `--max_grad_norm`              | Gradient clipping threshold                                                                           |
| `--learning-rate`              | Initial learning rate                                                                                 |
| `--gradient_accumulation_steps`| Steps to accumulate gradients (for larger batch sizes)                                                |
| `--max_steps`                  | Total training steps                                                                                  |
| `--dataset_name`               | Path to tokenized dataset                                                                             |
| `--batch-size`                 | Per-device batch size                                                                                 |
| `--data-max-len`               | Max input sequence length                                                                             |
| `--save_steps`                 | Save checkpoint every N steps                                                                         |
| `--check_data_cls_loss`        | Enable/disable custom classification loss                                                             |
| `--target_hidden_size`         | Student model’s hidden size (for distillation/custom arch)                                            |
| `--kl_temperature`             | Temperature for KL divergence (knowledge distillation)                                                |
| `--warmup-ratio`               | LR warmup ratio                                                                                       |
| `--raw-model-name`             | Path to teacher model                                                                                 |
| `--extra_tags`                 | Comma-separated experiment tags                                                                       |
| `--use_accelerate`             | Explicitly enable Accelerate features                                                                 |
| `--output_dir`                 | Checkpoint/output directory                                                                           |
| `--str_ban_losses`             | Specify losses to ignore (e.g., `mlp-gate-loss,attn-q-sim-loss`)                                     |
| `--tie_word_emb_proj`          | 1 to tie word embeddings and LM head                                                                 |
| `--use_all_attn`               | 1 for all attention layers, 0 otherwise                                                              |
| `--aux_loss_scale_factor`      | Scale for auxiliary (clone) loss                                                                     |

---

### ♻️ Checkpoint Conversion

Convert an LRC checkpoint into a Hugging Face-compatible student model:

```bash
python convert_ckpt.py \
  --ckpt-path CKPT_PATH \
  --target-hidden-size STUDENT_HIDDEN_SIZE \
  --raw-model-name TEACHER_PATH \
  --save-path STUDENT_SAVE_PATH \
  --use-all-attn ALL_ATTN_OR_NOT \
  --use-in-out-mlp USE_IO_FFN_OR_NOT \
  --tie-word-emb-proj TIE_WORD_EMB_OR_NOT
```

**Parameter Guide:**

| Argument              | Description                                                      | Example                                 |
| --------------------- | --------------------------------------------------------------- | --------------------------------------- |
| `--ckpt-path`         | Path to LRC checkpoint (`.safetensors`)                         | `../ckpts/lrc_model/model.safetensors`  |
| `--target-hidden-size`| Student model hidden size (**must match training**)              | `1536`                                  |
| `--raw-model-name`    | Teacher model path (for config/base weights)                    | `../models/Llama-3.2-3B-Instruct/`      |
| `--save-path`         | Output directory for student model                              | `../converted_models/student_model/`    |
| `--use-all-attn`      | 1 if "all attention" was used, else 0                           | `1` or `0`                              |
| `--use-in-out-mlp`    | 1 if "in/out MLP" (FFN) was used, else 0                        | `1` or `0`                              |
| `--tie-word-emb-proj` | 1 if word embeddings are tied with output projection, else 0    | `1` or `0`                              |

> **Remember:**  
> Replace all placeholder paths and flags with your real values, matching those used during LRC training!

---

### 🧑‍🎓 Supervised Fine-Tuning (SFT)

Before SFT, please convert the checkpoint. Then Fine-tune your model with LlamaFactory:

```bash
# Be sure to update your model checkpoint path in the YAML config!
FORCE_TORCHRUN=1 llamafactory-cli train ../low-rank-clone/configs/llama_factory/llama3-sft-full.yaml
```

**Notes:**
- `FORCE_TORCHRUN=1` is required for `llamafactory-cli`.
- Adjust `model_name_or_path` in your YAML config to point to your LRC-trained checkpoint.


## 3. Data Preparation

All datasets are in pre-tokenized `jsonl` format.

### 🛠️ Example: Data Generation

> **Note:**  
> The data paths in the script are hardcoded. Edit them as needed for your environment.

```bash
python data/generate_general_data_parallel.py \
  --version v5.1 \
  --tkn-path TEACHER_MODEL_PATH \
  --num-workers 8 \
  --data-max-len 2048
```

**Key Arguments:**
- `--version`: Data generation version (e.g., `v5.1`)
- `--tkn-path`: Path to teacher model tokenizer
- `--num-workers`: Number of parallel workers
- `--data-max-len`: Max sequence length per example

## 4. Evaluation

> **Required:**  
> Run evaluation **inside the `lm_eval` environment**.

```bash
lm_eval \
  --model hf \
  --tasks "sciq,piqa,winogrande,arc_easy,logiqa,arc_challenge,boolq,mmlu,commonsense_qa" \
  --batch_size "auto" \
  --trust_remote_code \
  --num_fewshot 0 \
  --model_args pretrained=YOUR_MODEL_PATH
```

**Evaluation Arguments:**
- `--model hf`: Use Hugging Face model interface
- `--tasks`: Comma-separated list of evaluation benchmarks
- `--batch_size "auto"`: Auto-select batch size
- `--trust_remote_code`: Enable custom model hub code
- `--num_fewshot 0`: Number of examples for few-shot (0 = zero-shot)
- `--model_args pretrained=YOUR_MODEL_PATH`: Path to your trained model checkpoint  
  *(replace `YOUR_MODEL_PATH` with your actual model directory)*

---

