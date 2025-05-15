import os
import json
from typing import List, Dict
from tqdm import tqdm
from random import shuffle
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset, IterableDataset, IterableDatasetDict
from transformers import LlamaTokenizer, AutoTokenizer


def merge_jsonl_files(folder_path: str) -> List[Dict]:
    """
    Recursively traverses a folder and merges the contents of all .json files (in JSONL format).

    Args:
        folder_path (str): Path to the folder to traverse

    Returns:
        List[Dict]: List of merged data
    """
    result = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    # Recursively traverse the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Read the JSONL file line by line
                        for line in f:
                            try:
                                # Parse each line of JSON data
                                data = json.loads(line.strip())
                                result.append(data)
                            except json.JSONDecodeError as e:
                                print(
                                    f"Error parsing JSON in file {file_path}, line: {line}"
                                )
                                print(f"Error message: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error reading file {file_path}")
                    print(f"Error message: {str(e)}")
                    continue

    return result


def get_any_dataset(dataset_name, tokenizer: LlamaTokenizer = None) -> DatasetDict:
    if "mix_general" in dataset_name or "mix_sft" in dataset_name:
        def gen():
            with open(dataset_name, "r", encoding="utf-8") as _in:
                for line in _in:
                    yield json.loads(line)
        train = IterableDataset.from_generator(gen)
        # Using iterable dataset requires specifying steps directly, which feels a bit silly
        # print(dataset_name, "size is", len(train))
        return IterableDatasetDict({"train": train})
    
    if "mix_" in dataset_name:  # Generated data
        data = load_from_disk(dataset_name)
        return DatasetDict({"train": data})

    if "ultrachat_200k" in dataset_name:
        data = load_dataset(dataset_name)
        assert tokenizer is not None, "Tokenizer required"
        lst = data["train_sft"].to_list()
        nlst = []
        for d in lst:
            conversations = d["messages"]
            text = tokenizer.apply_chat_template(conversations, tokenize=False)
            nlst += [{"text": text}]

        shuffle(nlst)
        train = Dataset.from_list(nlst)
        return DatasetDict({"train": train})

    if "squad" in dataset_name:
        data = load_from_disk(dataset_name)
        return data

    if "medical_big" in dataset_name:
        assert tokenizer is not None, "Tokenizer required"
        data = merge_jsonl_files(dataset_name)
        shuffle(data)
        data_new = []
        # force limit 
        for d in tqdm(data, desc="medical-big processing"):
            if "input" in d:  # fine tune data
                nd = {}
                # nd["text"] = f"{d['instruction']}\n{d['input']}\n{d['output']}"
                chat = [
                    {"role": "user", "content": f"{d['instruction']}\n{d['input']}"},
                    {"role": "assistant", "content": d['output']},
                ]
                nd["text"] = tokenizer.apply_chat_template(chat, tokenize=False)
                data_new += [nd]
                continue

            if "response_chosen" in d:
                continue

            if "text" in d:
                if not isinstance(d["text"], str):
                    continue
                data_new += [{"text": d["text"]}]

        print(dataset_name, "(Data Size)", len(data_new))
        ret = Dataset.from_list(data_new)
        ret = ret.shuffle(seed=218)
        return DatasetDict({"train": ret})
    
    if "guidelines" in dataset_name:
        def create_text_column(x):
            return {"text": x["clean_text"]}
        
        data = load_dataset(dataset_name)
        # print(dataset_name, "(Data Size)", len(data))
        ndata = data.map(create_text_column)
        # print(data.column_names)
        ndata = ndata.remove_columns(data["train"].column_names)
        ndata = ndata.shuffle(seed=218)
        return ndata
    
    if "redpajama" in dataset_name.lower():
        data = merge_jsonl_files(dataset_name)
        data = [{"text": d["text"]} for d in tqdm(data) if isinstance(d["text"], str)]
        print(dataset_name, "(After Filter Data Size)", len(data))
        shuffle(data)
        data = Dataset.from_list(data)
        return DatasetDict({"train": data})
    
    if "OpenHermes" in dataset_name:
        data = load_dataset(dataset_name)
        assert tokenizer is not None, "Tokenizer required"
        lst = data["train"].to_list()
        nlst = []
        for d in lst:
            conversations = d["conversations"]
            _new_conversations = []
            for _d in conversations:
                _new_conversations += [{
                    "role": _d["from"].replace("human", "user").replace("gpt", "assistant"),
                    "content": _d["value"],
                }]
            text = tokenizer.apply_chat_template(_new_conversations, tokenize=False)
            nlst += [{"text": text}]
        
        shuffle(nlst)
        train = Dataset.from_list(nlst)
        return DatasetDict({"train": train})
    
    if "pubmed_abs" in dataset_name:
        return load_dataset(dataset_name)
    
    if "meta_math" in dataset_name:
        assert tokenizer is not None, "Tokenizer required"
        data = load_dataset(dataset_name)
        lst = data["train"].to_list()
        nlst = []
        for _i, d in enumerate(lst):
            query, resp = d["query"], d["response"]
            if _i % 3 == 0:
                conversations = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": resp},
                ]
                text = tokenizer.apply_chat_template(conversations, tokenize=False)
            else:
                text = f"{query}\nAnswer: {resp}"
            nlst += [{"text": text}]

        train_set = Dataset.from_list(nlst)
        return DatasetDict({"train": train_set})

    if "pile-of-law" in dataset_name:
        raise NotImplementedError
    
    if "med_mcqa" in dataset_name:
        # question, exp, cop, opa, opb, ...
        def medmcqa_template(d):
            if d['exp'] is not None and len(d['exp']) > 0:
                out = f"Explanation: {d['exp']}\nAnswer: {d['cop']}"
            else:
                out = f"Answer: {d['cop']}"
            d = {
                "instruction": f"{d['question']}",
                "input": f"Options:\nA. {d['opa']}\nB. {d['opb']}\nC. {d['opc']}\n D. {d['opd']}",
                "output": out
            }
            return {"text": f"{d['instruction']}\n{d['input']}\nd{'output'}"}
        
        medmcqa = load_dataset("/root/autodl-tmp/datasets/med_mcqa")
        raw_cols = medmcqa["train"].column_names
        medmcqa = medmcqa.map(medmcqa_template, batched=False)
        print(raw_cols)
        medmcqa = medmcqa.remove_columns(raw_cols)
        raise NotImplementedError
    
    raise ValueError("Dataset Not Found!")


if __name__ == "__main__":

    # get_any_dataset("../datasets/medical_big")
    print(get_any_dataset("/root/autodl-tmp/datasets/RedPajama-Data-1T-Sample"))