from transformers import LlamaTokenizer
from tools.global_state import data_cls_dict
import torch

def get_any_tokenize_func(
    dataset_name: str, tokenizer: LlamaTokenizer, data_max_len=510
):
    dataset_name = dataset_name.lower()
    if "squad" in dataset_name:
        def tokenize_function(x):
            texts = [x["instruction"] + "\n" + x["output"] + tokenizer.eos_token]

            # Tokenize the concatenated text
            res = tokenizer(
                texts,
                padding="longest",
                max_length=data_max_len,  # Set maximum length
                truncation=True,  # Truncate if exceeding the maximum length
                return_tensors="pt",
            )
            # res["labels"] = res["input_ids"]
            return res

        return tokenize_function
    
    elif "tokenize" in dataset_name:
        def tokenize_function(x):
            if len(x["input_ids"]) > 1:
                x["input_ids"] = [x["input_ids"]]
            x["input_ids"][0] = x["input_ids"][0][:data_max_len]
            if "data_cls" in x:
                x["data_cls"] = data_cls_dict[x["data_cls"]]
            return x

        return tokenize_function
    
    else:
        def tokenize_function(x):
            texts = x["text"]
            # Tokenize the concatenated text
            res = tokenizer(
                texts,
                padding="longest",
                max_length=data_max_len,  # Set maximum length
                truncation=True,  # Truncate if exceeding the maximum length
                return_tensors="pt",
            )
            res["labels"] = res["input_ids"]
            return res

        return tokenize_function
    
    # else:
    #     raise ValueError


def get_any_data_collator(dataset_name: str, tokenizer: LlamaTokenizer, data_max_len=510):
    if "tokenize" in dataset_name:
        def data_collator(sample_lst):
            res = {"input_ids": []}
            for sample in sample_lst:
                res["input_ids"] += sample["input_ids"]
            if "data_cls" in sample_lst[0]:
                res["data_cls"] = []
                for sample in sample_lst:
                    res["data_cls"] += [sample["data_cls"]]
            
            res["input_ids"] = torch.tensor(res["input_ids"], dtype=torch.int64)
            res["labels"] = res["input_ids"]
            
            if "data_cls" in res:
                res["data_cls"] = torch.tensor(res["data_cls"], dtype=torch.int64)

            return res
        
        return data_collator
    
    else:
        def data_collator(batch_lis):
            res = {"input_ids": []}
            for sample in batch_lis:
                for k in res:
                    res[k] += sample[k]
            res = tokenizer.pad(res, return_tensors="pt")
            res["labels"] = res["input_ids"]
            return res
        
        return data_collator
