import torch
from transformers import AutoModelForCausalLM
from fire import Fire
from data.get_any_data import get_any_dataset
from tqdm import tqdm

def main(model_path, data_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    data = get_any_dataset(data_path)['train']
    cnt = 0
    loss = 0
    for line in tqdm(data):
        # print(line)
        ipt_ids = [line['input_ids']]
        ipt_ids = torch.tensor(ipt_ids).cuda(0)
        with torch.no_grad():
            out = model(input_ids=ipt_ids, labels=ipt_ids)
        # print(out.loss)
        loss += out.loss.item()

        cnt += 1
        if cnt == 2000:
            break

    print(loss / cnt)

if __name__ == "__main__":
    Fire(main)
