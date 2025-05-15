import json
import os
from tqdm import tqdm
from fire import Fire

def main(dataset_path):
    final_lst = []
    mm = 0
    with open(dataset_path, "r") as _in:
        for i, line in tqdm(enumerate(_in)):
            d = json.loads(line)
            max_v = max(d["input_ids"])
            mm = max(mm, max_v)
            if max_v >= 32000:
                print(i)
            else:
                final_lst.append(line)
    print(max_v)
    path = os.path.split(dataset_path)[0]
    with open(f"{path}/fixed.jsonl", 'w') as _out:
        print("writing")
        _out.writelines(final_lst)

if __name__ == "__main__":
    Fire(main)
