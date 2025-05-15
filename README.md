# Low-Rank Clone (LRC)

Code for the paper.

## 1. Environment Configuration

**Important:** Training LRC models requires `transformers` version 4.41.2 or lower.

### LRC Environment Setup

```shell
conda create -n lrc python=3.10 -y
conda activate lrc

pip install torch==2.3.0
pip install transformers[torch]==4.41.2
pip install deepspeed==0.15.4
pip install fire matplotlib seaborn datasets==2.19.2 datatrove
pip install wandb
pip install accelerate==1.1.1
MAX_JOBS=8 pip install flash-attn --no-build-isolation
````

### lm\_eval Environment Setup

```shell
conda create -n lm_eval --clone lrc
conda activate lm_eval

pip install transformers==4.51.3
pip install lm_eval==0.4.8
```

### LlamaFactory Environment Setup

```shell
conda create -n lf --clone lm_eval
conda activate lf

# Navigate to your LlamaFactory directory
cd llamafactory_PATH 

# Install LlamaFactory (ensure you have the necessary install scripts/commands for LlamaFactory)
```

## 2\. Running Code Examples

### LRC-1.5B Training Script

The following script is used for training the LRC-1.5B model:

```shell
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

#### Parameter Explanations

Below is an explanation of the parameters used in the LRC-1.5B training script. This information is based on common practices and documentation for Hugging Face Accelerate and Trainer.

| Parameter                       | Description                                                                                                                               |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `accelerate launch`             | Command to launch a script for distributed training using Hugging Face Accelerate.                                               |
| `--main_process_port`           | The port number for the main process in distributed training.                                                                      |
| `--config_file`                 | Path to the Accelerate configuration file (e.g., `configs/accel_ds_8h800_gas1.yaml`). This file contains settings for distributed training. |
| `hf_trainer.py`                 | The Python script used for training, based on Hugging Face Trainer.                                                                |
| `--log_steps`                   | Log training metrics every N steps.                                                                                                |
| `--max_grad_norm`               | Maximum gradient norm for gradient clipping. This helps prevent exploding gradients.                                                           |
| `--learning-rate`               | The initial learning rate for the optimizer.                                                                                                |
| `--gradient_accumulation_steps` | Number of steps to accumulate gradients before performing an optimizer step. This is useful for training with larger effective batch sizes.         |
| `--max_steps`                   | Total number of training steps to perform.                                                                                                |
| `--dataset_name`                | Path to the training dataset file (e.g., `../datasets/mix_general_llama3_tokenized_v5.1/train.jsonl`).                                     |
| `--batch-size`                  | Batch size per device for training.                                                                                                       |
| `--data-max-len`                | Maximum sequence length for the input data.                                                                                               |
| `--save_steps`                  | Save a model checkpoint every N steps.                                                                                              |
| `--check_data_cls_loss`         | A custom boolean flag to enable or disable checking a specific classification loss related to data.                                     |
| `--target_hidden_size`          | The target hidden size for the student model, relevant for model distillation or a custom architecture.                                      |
| `--kl_temperature`              | Temperature parameter for KL divergence loss, used in knowledge distillation.                                                       |
| `--warmup-ratio`                | Ratio of total training steps used for a linear warmup of the learning rate.                                                              |
| `--raw-model-name`              | Path to the base or teacher model (e.g., `TEACHER_MODEL_PATH`).                                                            |
| `--extra_tags`                  | Custom tags for experiment tracking (e.g., with WandB). These are used to categorize or identify runs.                                               |
| `--use_accelerate`              | A custom boolean flag to explicitly enable or disable features related to Hugging Face Accelerate.                                        |
| `--output_dir`                  | Directory to save model checkpoints and other outputs.                                                                                    |
| `--str_ban_losses`              | A custom string flag to specify certain losses to ignore during training (Keys: `mlp-gate-loss,mlp-up-loss,mlp-out-loss,attn-q-sim-loss,attn-k-sim-loss,attn-v-sim-loss,attn-out-sim-loss`).                                                       |
| `--tie_word_emb_proj`           | A custom flag (0 or 1) to indicate whether to tie word embeddings with lm head projection layer.                             |
| `--use_all_attn`                | A custom flag (0 or 1) to use `all attn` or `io attn`                                         |
| `--use_in_out_mlp`                | A custom flag (0 or 1) to use `all ffn` or `io ffn`                                         |
| `--aux_loss_scale_factor`       | Scaling factor for clone loss        |

### SFT (Supervised Fine-Tuning)

```shell
# Remember to modify your model checkpoint path in the YAML file
FORCE_TORCHRUN=1 llamafactory-cli train ../low-rank-clone/configs/llama_factory/llama3-sft-full.yaml
```

**Note:**

  * `FORCE_TORCHRUN=1` is an environment variable that is required for `llamafactory-cli`.
  * The `llamafactory-cli train` command takes a YAML configuration file that specifies the parameters for fine-tuning.
  * You **must modify the `model_name_or_path`** within the YAML file (`../low-rank-clone/configs/llama_factory/llama3-sft-full.yaml`) to point to your trained model checkpoint.

### Convert Checkpoint

This script converts a trained LRC checkpoint into a standard Hugging Face model format, merging weights and adjusting the configuration for a smaller "student" model.

**Usage:**

```shell
python convert_ckpt.py \
--ckpt-path CKPT_PATH \
--target-hidden-size STUDENT_HIDDEN_SIZE \
--raw-model-name TEACHER_PATH \
--save-path STUDENT_SAVE_PATH \
--use-all-attn ALL_ATTN_OR_NOT \
--use-in-out-mlp USE_IO_FFN_OR_NOT \
--tie-word-emb-proj TIE_WORD_EMB_OR_NOT
```

**Parameter Explanations:**

| Parameter              | Description                                                                                                   | Example Value                    |
| :--------------------- | :------------------------------------------------------------------------------------------------------------ | :------------------------------- |
| `--ckpt-path`          | Path to the trained Low-Rank Clone (LRC) checkpoint file (a `.safetensors` file).                       | `../ckpts/lrc_model/model.safetensors` |
| `--target-hidden-size` | The hidden size of the target student model. This **must** match the `target_hidden_size` used during training. | `1536`                           |
| `--raw-model-name`     | Path to the original teacher model. This is used to load the base configuration.                             | `../models/Llama-3.2-3B-Instruct/` |
| `--save-path`          | Directory where the converted student model (Hugging Face format) will be saved.                            | `../converted_models/student_model/` |
| `--use-all-attn`       | Boolean flag (0 or 1) indicating if the "all attention" configuration was used during training.               | `1` (True) or `0` (False)        |
| `--use-in-out-mlp`     | Boolean flag (0 or 1) indicating if the "input/output MLP" (FFN) configuration was used during training.       | `0` (False) or `1` (True)        |
| `--tie-word-emb-proj`  | Boolean flag (0 or 1) indicating if word embeddings were tied with the output projection during training.      | `1` (True) or `0` (False)        |

**Note:** Replace placeholder paths and values (e.g., `CKPT_PATH`, `TEACHER_PATH`, `STUDENT_SAVE_PATH`, `STUDENT_HIDDEN_SIZE`) with your actual values. The boolean flags must be set to match the configuration used during the LRC model training.

## 3\. Data

The data used for this project is in `jsonl` format and has already been tokenized.

### Data Generation Example (in `lrc` environment)

**Important:** The data paths in the script are hardcoded. You will need to modify them according to your local setup.

```shell
python data/generate_general_data_parallel.py \
--version v5.1 \
--tkn-path TEACHER_MODEL_PATH \
--num-workers 8 \
--data-max-len 2048
```

**Parameters for `generate_general_data_parallel.py`:**

  * `--version`: Specifies the version of the data generation process (e.g., `v5.1`).
  * `--tkn-path`: Path to the tokenizer of the teacher model (e.g., `TEACHER_MODEL_PATH`).
  * `--num-workers`: Number of worker processes to use for parallel data generation.
  * `--data-max-len`: Maximum sequence length for the generated data.

## 4\. Evaluate

**MUST\!** Perform evaluation in the `lm_eval` environment.

```shell
lm_eval \
    --model hf \
    --tasks "sciq,piqa,winogrande,arc_easy,logiqa,arc_challenge,boolq,mmlu,commonsense_qa" \
    --batch_size "auto" \
    --trust_remote_code \
    --num_fewshot 0 \
    --model_args pretrained=YOUR_MODEL_PATH
```

**Parameters for `lm_eval`:**

  * `--model hf`: Specifies that a Hugging Face model will be evaluated.
  * `--tasks`: A comma-separated list of evaluation tasks to perform.
  * `--batch_size "auto"`: Automatically determines the batch size for evaluation.
  * `--trust_remote_code`: Allows execution of custom code from the model hub.
  * `--num_fewshot 0`: Number of few-shot examples to provide in the context for each task (0 for zero-shot).
  * `--model_args pretrained=YOUR_MODEL_PATH`: Specifies the arguments for loading the model, where `pretrained` is the path to your trained model checkpoint. **Remember to replace `YOUR_MODEL_PATH` with the actual path.**
