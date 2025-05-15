#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: $0 <models_folder_path> <task>"
    echo "  <models_folder_path>: Path to the directory containing model folders"
    echo "  <task>: Task to evaluate models on"
    exit 1
}

# Check if correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi

MODELS_FOLDER="$1"
TASK="$2"

# Check if models folder exists
if [ ! -d "$MODELS_FOLDER" ]; then
    echo "Error: Models folder '$MODELS_FOLDER' does not exist."
    exit 1
fi

# Check if evaluation script exists
if [ ! -f "scripts/eval.sh" ]; then
    echo "Error: Evaluation script 'scripts/eval.sh' not found."
    exit 1
fi

# Find all model directories in the specified folder
echo "Finding models in $MODELS_FOLDER..."
MODEL_COUNT=0

# Process each directory in the models folder
for MODEL_PATH in "$MODELS_FOLDER"/*; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_NAME=$(basename "$MODEL_PATH")
        echo "----------------------------------------"
        echo "Evaluating model: $MODEL_NAME"
        echo "Model path: $MODEL_PATH"
        echo "Task: $TASK"
        echo "----------------------------------------"

        # Run the evaluation script
        bash scripts/eval.sh "$MODEL_PATH" "$TASK"

        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            echo "Evaluation completed for $MODEL_NAME"
        else
            echo "Warning: Evaluation may have failed for $MODEL_NAME"
        fi

        MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
done

if [ $MODEL_COUNT -eq 0 ]; then
    echo "No model directories found in '$MODELS_FOLDER'."
    exit 1
else
    echo "----------------------------------------"
    echo "Evaluation completed for $MODEL_COUNT models."
    echo "----------------------------------------"
fi