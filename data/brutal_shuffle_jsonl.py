import fire
import json
import random
from tqdm import tqdm

def shuffle_jsonl(input_path, output_path, seed=218):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Read the JSONL file
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = [json.loads(line) for line in tqdm(infile, desc="Reading", unit=" lines")]

    # Shuffle the data
    random.shuffle(data)

    # Write the shuffled data back to a new JSONL file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in tqdm(data, desc="Writing", unit=" lines"):
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_path, output_path):
    shuffle_jsonl(input_path, output_path)

if __name__ == '__main__':
    fire.Fire(main)
