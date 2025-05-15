import torch
import random
import json
import os
import math
import subprocess
import fire

from torch.utils.data import IterableDataset, DataLoader
from datasets import IterableDataset as HfIterableDataset
from transformers import AutoTokenizer, set_seed
from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm
from get_any_data import get_any_dataset

set_seed(218)
random.seed(218)

class ShardedDataset(IterableDataset):
    def __init__(self, data_source, data_cls, total_max_items, data_max_len, tokenizer, skip_factor=1, min_edu_score=0):
        self.data_source = data_source
        self.data_cls = data_cls
        self.total_max_items = total_max_items
        self.data_max_len = data_max_len
        self.tokenizer = tokenizer
        self.skip_factor = 1.0 / skip_factor  # Keep 1/skip_factor of the data
        self.min_edu_score = min_edu_score

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single process
            world_size = 1
            rank = 0
        else:
            world_size = worker_info.num_workers
            rank = worker_info.id

        # Calculate max_items for each worker
        if self.total_max_items:
            remainder = self.total_max_items % world_size
            per_worker_max_items = self.total_max_items // world_size
            if rank < remainder:
                per_worker_max_items += 1
        else:
            per_worker_max_items = None

        buffer = []
        cnt = 0
        eos_token = self.tokenizer.eos_token
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""
        for i, sample in enumerate(self.data_source):
            # Sharding logic: does this sample belong to the current worker?
            if i % world_size != rank:
                continue

            if self.skip_factor < 1.0 - 1e-6 and random.random() > self.skip_factor:
                continue
            # Process sample

            _edu_score = 999
            if isinstance(sample, dict):
                text = eos_token + sample["text"] + bos_token
                if "metadata" in sample:
                    _edu_score = sample["metadata"].get("int_score", 5)
            else:
                text = eos_token + sample.text + bos_token
                if hasattr(sample, "metadata"):
                    _edu_score = sample.metadata.get("int_score", 5)

            if _edu_score < self.min_edu_score:
                continue

            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            buffer.append(tokens)
            if len(buffer) > 1:
                buffer = [torch.cat(buffer)]
            while len(buffer[0]) >= self.data_max_len:
                chunk = buffer[0][:self.data_max_len]
                yield json.dumps({
                    "input_ids": chunk.tolist(),
                    "data_cls": self.data_cls
                }, ensure_ascii=False) + "\n"
                cnt += 1
                if per_worker_max_items and cnt >= per_worker_max_items:
                    return
                buffer[0] = buffer[0][self.data_max_len:]

def get_dataloader(dataset, num_workers):
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )

def shuffle_jsonl(input_path, output_path):
    """
    Use Linux `shuf` command to shuffle the input JSONL file and save to output path
    :param input_path: Input JSONL file path
    :param output_path: Output JSONL file path
    """
    try:
        # Use subprocess to call `shuf` command to shuffle the file
        with open(output_path, 'w') as outfile:
            subprocess.run(['shuf', input_path], stdout=outfile, check=True)
        print(f"Shuffled file saved to {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

# =================== Data processing functions for each Version ===================

def process_sft_v1(model_cls, tkn, data_max_len, num_workers):
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_sft_v1.0"
    os.makedirs(output_path, exist_ok=True)

    data_hermes = get_any_dataset("../xxxx/datasets/ultrachat_200k", tkn)["train"]
    data_hermes = get_dataloader(ShardedDataset(data_hermes, "general", None, data_max_len, tkn, skip_factor=1), num_workers)
    with open(f"{output_path}/ultrachat.jsonl", "w", encoding="utf-8") as _o:
        _cnt = 0
        for line in tqdm(data_hermes, desc="ultrachat"):
            _cnt += 1
            _o.write(line)
        print(f"in total, ultrachat lines: {_cnt}")
    shuffle_jsonl(f"{output_path}/ultrachat.jsonl", f"{output_path}/ultrachat.shuffled.jsonl")

def process_v5_1_20b(model_cls, tkn, data_max_len, num_workers):
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v5.1.20b"
    os.makedirs(output_path, exist_ok=True)

    in_paths = [
        "../xxxx/datasets/fineweb-edu-dedup",
    ]
    out_names = [
        "fineweb-edu-dedup",
    ]
    skip_factors = [
        1,
    ]
    for in_path, skip_factor, out_name in zip(in_paths, skip_factors, out_names):
        data = ParquetReader(in_path)()
        data = get_dataloader(
            ShardedDataset(data, "general", None, data_max_len, tkn,
                           skip_factor=skip_factor, min_edu_score=4),
            num_workers
        )
        with open(f"{output_path}/{out_name}.jsonl", "w", encoding="utf-8") as _o:
            _cnt = 0
            for line in tqdm(data, desc=out_name):
                _cnt += 1
                _o.write(line)
            print(f"in total, {out_name} lines: {_cnt}")
        shuffle_jsonl(f"{output_path}/{out_name}.jsonl", f"{output_path}/{out_name}.shuffled.jsonl")

    data_hermes = get_any_dataset("../xxxx/datasets/OpenHermes-2.5", tkn)["train"]
    data_hermes = get_dataloader(ShardedDataset(data_hermes, "general", None, data_max_len, tkn, skip_factor=1), 1)
    with open(f"{output_path}/hermes.jsonl", "w", encoding="utf-8") as _o:
        _cnt = 0
        for line in tqdm(data_hermes, desc="hermes"):
            _cnt += 1
            _o.write(line)
        print(f"in total, hermes lines: {_cnt}")
    shuffle_jsonl(f"{output_path}/hermes.jsonl", f"{output_path}/hermes.shuffled.jsonl")

    data_names = [
        "fineweb-edu-dedup",
        "hermes",
    ]

    limit_tokens = [20e9, 1e9]
    with open(f"{output_path}/no_shuffle.jsonl", 'w', encoding='utf-8') as _out:
        for in_file, limit in zip(data_names, limit_tokens):
            with open(f"{output_path}/{in_file}.shuffled.jsonl", "r") as _in:
                limit //= data_max_len
                cnt = 0
                for line in tqdm(_in, total=limit):
                    _out.write(line)
                    cnt += 1
                    if cnt >= limit:
                        print(f"{in_file} reached the specified limit {limit}")
                        break
                print(f"{in_file} ended cnt = {cnt}")

    shuffle_jsonl(
        f"{output_path}/no_shuffle.jsonl",
        f"{output_path}/train.jsonl"
    )

def process_v5_1(model_cls, tkn, data_max_len, num_workers):
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v5.1"
    os.makedirs(output_path, exist_ok=True)

    in_paths = [
        "../xxxx/datasets/fineweb-edu-dedup",
    ]
    out_names = [
        "fineweb-edu-dedup",
    ]
    skip_factors = [
        1,
    ]
    for in_path, skip_factor, out_name in zip(in_paths, skip_factors, out_names):
        data = ParquetReader(in_path)()
        data = get_dataloader(
            ShardedDataset(data, "general", None, data_max_len, tkn,
                           skip_factor=skip_factor, min_edu_score=4),
            num_workers
        )
        with open(f"{output_path}/{out_name}.jsonl", "w", encoding="utf-8") as _o:
            _cnt = 0
            for line in tqdm(data, desc=out_name):
                _cnt += 1
                _o.write(line)
            print(f"in total, {out_name} lines: {_cnt}")
        shuffle_jsonl(f"{output_path}/{out_name}.jsonl", f"{output_path}/{out_name}.shuffled.jsonl")

    data_hermes = get_any_dataset("../xxxx/datasets/OpenHermes-2.5", tkn)["train"]
    data_hermes = get_dataloader(ShardedDataset(data_hermes, "general", None, data_max_len, tkn, skip_factor=1), 1)
    with open(f"{output_path}/hermes.jsonl", "w", encoding="utf-8") as _o:
        _cnt = 0
        for line in tqdm(data_hermes, desc="hermes"):
            _cnt += 1
            _o.write(line)
        print(f"in total, hermes lines: {_cnt}")
    shuffle_jsonl(f"{output_path}/hermes.jsonl", f"{output_path}/hermes.shuffled.jsonl")

    data_names = [
        "fineweb-edu-dedup",
        "hermes",
    ]

    limit_tokens = [10e9, 1e9]
    with open(f"{output_path}/no_shuffle.jsonl", 'w', encoding='utf-8') as _out:
        for in_file, limit in zip(data_names, limit_tokens):
            with open(f"{output_path}/{in_file}.shuffled.jsonl", "r") as _in:
                limit //= data_max_len
                cnt = 0
                for line in tqdm(_in, total=limit):
                    _out.write(line)
                    cnt += 1
                    if cnt >= limit:
                        print(f"{in_file} reached the specified limit {limit}")
                        break
                print(f"{in_file} ended cnt = {cnt}")

    shuffle_jsonl(
        f"{output_path}/no_shuffle.jsonl",
        f"{output_path}/train.jsonl"
    )

def process_v5_0(model_cls, tkn, data_max_len, num_workers):
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v5.0"
    os.makedirs(output_path, exist_ok=True)

    in_paths = [
        # "../xxxx/datasets/smollm-corpus/cosmopedia-v2",
        "../xxxx/datasets/dclm-baseline-1.0-parquet",
        # "../xxxx/datasets/fineweb-edu-dedup",
    ]
    out_names = [
        # "cosmo",
        "dclm",
        # "fineweb-edu-dedup",
    ]
    skip_factors = [
        # 7, 
        2, 
        # 9,
    ]
    for in_path, skip_factor, out_name in zip(in_paths, skip_factors, out_names):
        data = ParquetReader(in_path)()
        data = get_dataloader(ShardedDataset(data, "general", None, data_max_len, tkn, skip_factor=skip_factor), num_workers)
        with open(f"{output_path}/{out_name}.jsonl", "w", encoding="utf-8") as _o:
            _cnt = 0
            for line in tqdm(data, desc=out_name):
                _cnt += 1
                _o.write(line)
            print(f"in total, {out_name} lines: {_cnt}")
        shuffle_jsonl(f"{output_path}/{out_name}.jsonl", f"{output_path}/{out_name}.shuffled.jsonl")
    
    # data_hermes = get_any_dataset("../xxxx/datasets/OpenHermes-2.5", tkn)["train"]
    # data_hermes = get_dataloader(ShardedDataset(data_hermes, "general", None, data_max_len, tkn, skip_factor=1), 1)
    # with open(f"{output_path}/hermes.jsonl", "w", encoding="utf-8") as _o:
    #     _cnt = 0
    #     for line in tqdm(data_hermes, desc="hermes"):
    #         _cnt += 1
    #         _o.write(line)
    #     print(f"in total, hermes lines: {_cnt}")
    # shuffle_jsonl(f"{output_path}/hermes.jsonl", f"{output_path}/hermes.shuffled.jsonl")

    data_names = [
        "cosmo",
        "dclm",
        "fineweb-edu-dedup",
        "hermes",
    ]

    limit_tokens = [1e9, 2e9, 18e9, 1e9]
    with open(f"{output_path}/no_shuffle.jsonl", 'w', encoding='utf-8') as _out:
        for in_file, limit in zip(data_names, limit_tokens):
            with open(f"{output_path}/{in_file}.shuffled.jsonl", "r") as _in:
                limit //= data_max_len
                cnt = 0
                for line in tqdm(_in, total=limit):
                    _out.write(line)
                    cnt += 1
                    if cnt >= limit:
                        print(f"{in_file} reached the specified limit {limit}")
                        break
                print(f"{in_file} ended cnt = {cnt}")
    
    shuffle_jsonl(
        f"{output_path}/no_shuffle.jsonl",
        f"{output_path}/train.jsonl"
    )

def process_v4_1(model_cls, tkn, data_max_len, num_workers):
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v4.1"
    os.makedirs(output_path, exist_ok=True)

    files = [
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/fineweb.shuffled.jsonl", 
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v4.0/dclm.shuffled.jsonl",             
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/cosmopedia.shuffled.jsonl"
    ]
    limits = [int(5e6), int(5e6), int(10e6)]
    with open(f"{output_path}/no_shuffle.jsonl", 'w', encoding='utf-8') as _out:
        for in_file, limit in zip(files, limits):
            with open(f"{in_file}", "r") as _in:
                cnt = 0
                for line in tqdm(_in, total=limit):
                    _out.write(line)
                    cnt += 1
                    if cnt == limit:
                        break
    
    shuffle_jsonl(
        f"{output_path}/no_shuffle.jsonl",
        f"{output_path}/train.jsonl"
    )

    with open(f"{output_path}/train.jsonl", "a", encoding="utf-8") as _out:
        with open(f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/hermes.shuffled.jsonl", "r") as _in:
            for line in tqdm(_in):
                _out.write(line)

    tokens_10b = int(10e6)
    cnt = 0
    with open(f"{output_path}/train.10b.jsonl", "w") as _out:
        with open(f"{output_path}/train.jsonl", "r") as _in:
            for line in tqdm(_in, total=tokens_10b, desc="sub 10b"):
                _out.write(line)
                cnt += 1
                if cnt == tokens_10b:
                    break
        with open(f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/hermes.shuffled.jsonl", "r") as _in:
            for line in tqdm(_in):
                _out.write(line)

def process_v4_0(model_cls, tkn, data_max_len, num_workers):
    data_dclm = ParquetReader("../xxxx/datasets/dclm-baseline-1.0-parquet")()
    data_dclm = get_dataloader(ShardedDataset(data_dclm, "general", None, data_max_len, tkn, 1), num_workers)
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v4.0"
    os.makedirs(output_path, exist_ok=True)

    files = [
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/fineweb.shuffled.jsonl", 
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v4.0/dclm.shuffled.jsonl",             
        f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/cosmopedia.shuffled.jsonl"
    ]
    limits = [int(10e6), int(6.6e6), int(3.4e6)]
    with open(f"{output_path}/no_shuffle.jsonl", 'w', encoding='utf-8') as _out:
        for in_file, limit in zip(files, limits):
            with open(f"{in_file}", "r") as _in:
                cnt = 0
                for line in tqdm(_in, total=limit):
                    _out.write(line)
                    cnt += 1
                    if cnt == limit:
                        break
    
    shuffle_jsonl(
        f"{output_path}/no_shuffle.jsonl",
        f"{output_path}/train.jsonl"
    )

    with open(f"{output_path}/train.jsonl", "a", encoding="utf-8") as _out:
        with open(f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/hermes.shuffled.jsonl", "r") as _in:
            for line in tqdm(_in):
                _out.write(line)

    tokens_10b = int(10e6)
    cnt = 0
    with open(f"{output_path}/train.10b.jsonl", "w") as _out:
        with open(f"{output_path}/train.jsonl", "r") as _in:
            for line in tqdm(_in, total=tokens_10b, desc="sub 10b"):
                _out.write(line)
                cnt += 1
                if cnt == tokens_10b:
                    break
        with open(f"../xxxx/datasets/mix_general_{model_cls}_tokenized_v3.1/hermes.shuffled.jsonl", "r") as _in:
            for line in tqdm(_in):
                _out.write(line)

def process_v3_1(model_cls, tkn, data_max_len, num_workers):
    data_cosmopedia_v2 = ParquetReader("../datasets/smollm-corpus/cosmopedia-v2")() # ~28B
    data_fineweb_edu = ParquetReader("../datasets/fineweb-edu-10B/sample/10BT")()
    data_hermes = get_any_dataset("../datasets/OpenHermes-2.5", tkn)["train"]
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_v3.1"
    os.makedirs(output_path, exist_ok=True)

    data_lst = [data_cosmopedia_v2, data_fineweb_edu, data_hermes]
    limits = [int(10e6), int(10e6), None]
    data_cls = ["general", "general", "sft"]
    tmp_file_name = ["cosmopedia", "fineweb", "hermes"]
    skip_factors = [2, 1, 1]

    a_cnt = 0
    shuffled_lst = []
    for data, limit, cls, tmp_file, skip in zip(data_lst, limits, data_cls, tmp_file_name, skip_factors):
        _data = get_dataloader(ShardedDataset(data, cls, None, data_max_len, tkn, skip), num_workers)
        with open(f"{output_path}/{tmp_file}.jsonl", "w", encoding="utf-8") as _out:
            for item in tqdm(_data):
                _out.write(item)
                a_cnt += 1
        shuffle_jsonl(f"{output_path}/{tmp_file}.jsonl", f"{output_path}/{tmp_file}.shuffled.jsonl")
        shuffled_lst += [f"{output_path}/{tmp_file}.shuffled.jsonl"]
    
    with open(f"{output_path}/no_shuffle.jsonl", "w", encoding="utf-8") as _out:
        for limit, file in zip(limits, shuffled_lst):
            with open(file, "r", encoding="utf-8") as _in:
                for i, line in tqdm(enumerate(_in), total=limit, desc="write"):
                    if limit is None or i <= limit:
                        _out.write(line)
                    else:
                        break
    
    shuffle_jsonl(f"{output_path}/no_shuffle.jsonl", f"{output_path}/train.jsonl")

def process_v2_4(model_cls, tkn, data_max_len, num_workers):
    data_fineweb_edu = ParquetReader("../datasets/fineweb-edu-10B/sample/10BT")()
    data_hermes = get_any_dataset("../datasets/OpenHermes-2.5", tkn)["train"]
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_shuffled_v2.4"
    if data_max_len != 1024:
        output_path += f"seqlen{data_max_len}"
    os.makedirs(output_path, exist_ok=True)

    # Create sharded datasets and process
    datasets = [
        get_dataloader(ShardedDataset(data_fineweb_edu, "general", None, data_max_len, tkn), num_workers), # 10B tokens
        get_dataloader(ShardedDataset(data_hermes, "sft", None, data_max_len, tkn), num_workers)  # ~429M tokens
    ]
    
    cnt = 0
    with open(f"{output_path}/no_shuffle.jsonl", "w", encoding="utf-8") as _out:
        for _data in datasets:
            for item in tqdm(_data):
                _out.write(item)
    
    shuffle_jsonl(f"{output_path}/no_shuffle.jsonl", f"{output_path}/train.jsonl")
    
    print(f"Has {cnt / 1e6}B tokens")  # 10.852B
    print(f"Has {cnt} data samples")

def process_v2_3(model_cls, tkn, data_max_len, num_workers):
    data_fineweb_edu = ParquetReader("/home/work/datasets/fineweb-edu-10B/sample/10BT")()
    data_hermes = get_any_dataset("/home/work/datasets/OpenHermes-2.5", tkn)["train"]
    output_path = f"/home/work/datasets/mix_general_{model_cls}_tokenized_v2.3"

    # Create sharded datasets and process
    datasets = [
        ShardedDataset(data_fineweb_edu, "general", None, data_max_len, tkn), # 10B tokens
        ShardedDataset(data_hermes, "sft", None, data_max_len, tkn)  # ~429M tokens
    ]

    iter0 = iter(get_dataloader(datasets[0], num_workers))
    iter1 = iter(get_dataloader(datasets[1], 10))
    
    cnt = 0
    empty_general_flag = False
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/train.jsonl", 'w', encoding="utf-8") as _out:
        while True:
            cnt += 1
            if cnt % 30 == 0:
                item = next(iter1, None)
                if item is None:
                    assert empty_general_flag
                    break
                _out.write(item)
            else:
                item = next(iter0, None)
                if item is None:
                    empty_general_flag = True
                else:
                    _out.write(item) 
            if cnt % 1000 == 0:
                print(f" processed {cnt} items")
    
    print(f"Has {cnt / 1e6}B tokens")  # 10.852B
    print(f"Has {cnt} data samples")

def process_redpajama1_0(model_cls, tkn, data_max_len, num_workers):
    data_redpajama = get_any_dataset("../datasets/redpajama", tkn)["train"]
    output_path = f"../datasets/mix_general_{model_cls}_tokenized_shuffled_redpajama1.0"
    os.makedirs(output_path, exist_ok=True)

    # Create sharded datasets and process
    datasets = [
        get_dataloader(ShardedDataset(data_redpajama, "general", None, data_max_len, tkn), num_workers),
    ]
    
    cnt = 0
    final_lst = []
    for _data in datasets:
        for item in tqdm(_data):
            final_lst.append(item)
            cnt += 1
    
    random.shuffle(final_lst)
    with open(f"{output_path}/train.jsonl", "w", encoding="utf-8") as _out:
        _out.writelines(final_lst)
    
    print(f"Has {cnt / 1e6}B tokens")
    print(f"Has {cnt} data samples")

# =================== Main function ===================

def main(
    version,
    tkn_path="../xxxx/models/Meta-Llama-3-8B-Instruct",
    num_workers=2,
    data_max_len=1024,
):
    tkn = AutoTokenizer.from_pretrained(tkn_path, use_fast=True)
    print("bos", tkn.bos_token)
    print("eos", tkn.eos_token)
    model_cls = None
    if "llama-3" in tkn_path.lower():
        model_cls = "llama3"
    elif "llama-2" in tkn_path.lower():
        model_cls = "llama2"
    elif "qwen" in tkn_path.lower():
        model_cls = "qwen"
    elif "phi" in tkn_path.lower():
        model_cls = "phi"
    else:
        raise ValueError

    # Call different processing functions according to version
    if version == "sft_v1.0":
        process_sft_v1(model_cls, tkn, data_max_len, num_workers)
    elif version == "v5.1.20b":
        process_v5_1_20b(model_cls, tkn, data_max_len, num_workers)
    elif version == "v5.1":
        process_v5_1(model_cls, tkn, data_max_len, num_workers)
    elif version == "v5.0":
        process_v5_0(model_cls, tkn, data_max_len, num_workers)
    elif version == "v4.1":
        process_v4_1(model_cls, tkn, data_max_len, num_workers)
    elif version == "v4.0":
        process_v4_0(model_cls, tkn, data_max_len, num_workers)
    elif version == "v3.1":
        process_v3_1(model_cls, tkn, data_max_len, num_workers)
    elif version == "v2.4":
        process_v2_4(model_cls, tkn, data_max_len, num_workers)
    elif version == "v2.3":
        process_v2_3(model_cls, tkn, data_max_len, num_workers)
    elif version == "redpajama1.0":
        process_redpajama1_0(model_cls, tkn, data_max_len, num_workers)
    else:
        raise ValueError(f"Unsupported version: {version}")

if __name__ == "__main__":
    fire.Fire(main)