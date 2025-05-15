#!/bin/bash

# Ensure a model path is provided
if [ -z "$1" ]; then
  echo "Usage: ./eval.sh <model_path> [task1,task2,...|all]"
  exit 1
fi

MODEL_PATH=$1
TASK=$2  # Optional: specify tasks as comma-separated values or "all" to run all tasks

# Define the tasks, their few-shot values, and batch sizes
declare -A TASKS
TASKS=(
  ["sciq"]=0
  ["piqa"]=0
  ["winogrande"]=0
  ["arc_easy"]=0
  ["logiqa"]=0
  ["arc_challenge"]=0
  ["hellaswag"]=0
  ["boolq"]=0
  ["mmlu"]=0
  ["commonsense_qa"]=0
  ["openbookqa"]=0
  ["truthfulqa"]=0
  # ["lambada_openai"]=0
  # ["arc_challenge"]=5
  # ["hellaswag"]=5
  # ["boolq"]=5
  # ["nq_open"]=5
  # ["mmlu"]=3
  # ["mmlu_continuation"]=0
)

declare -A BATCH_SIZES
BATCH_SIZES=(
  ["sciq"]=32
  ["piqa"]=32
  ["winogrande"]=32
  ["arc_easy"]=32
  ["logiqa"]=32
  # ["lambada_openai"]=32
  ["arc_challenge"]=32
  ["hellaswag"]=32
  ["boolq"]=8
  # ["nq_open"]=32
  ["mmlu"]=8
  # ["mmlu_continuation"]=32
  ["commonsense_qa"]=8
  ["openbookqa"]=32
  ["truthfulqa"]=32
)

# Function to run evaluation for a specific task
run_task() {
  local task_name=$1
  local fewshot=${TASKS[$task_name]}
  local batch_size=${BATCH_SIZES[$task_name]}

  echo "Running evaluation for task: $task_name with $fewshot few-shot and batch size $batch_size"

  lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH" \
    --tasks "$task_name" \
    --device cuda:0 \
    --batch_size "$batch_size" \
    --trust_remote_code \
    --num_fewshot "$fewshot"
}

# Check if "all" is specified or no task is provided, and run all tasks
if [ "$TASK" == "all" ] || [ -z "$TASK" ]; then
  for task in "${!TASKS[@]}"; do
    run_task "$task"
  done
else
  # Split the TASK string by comma into an array
  IFS=',' read -ra TASK_ARRAY <<< "$TASK"

  # Iterate through each specified task and validate it
  for task in "${TASK_ARRAY[@]}"; do
    if [[ -v TASKS[$task] ]]; then
      run_task "$task"
    else
      echo "Error: Task '$task' is not defined. Available tasks are: ${!TASKS[@]}"
      exit 1
    fi
  done
fi
