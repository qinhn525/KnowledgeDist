from transformers import (
    BertConfig, 
    AutoTokenizer, 
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    AdamW
)
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import json, numpy as np, pandas as pd, wandb, time

def load_model(path):
    config = BertConfig.from_pretrained(path)
    model = BertForMaskedLM.from_pretrained(path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_datasets(tokenizer):
    config = json.load(open("/home/qhn/Codes/Projects/KnowledgeDist/Data/mask/config.json", "r"))
    areas = list(config.keys())
    data_dir = "/home/qhn/Codes/Projects/KnowledgeDist/Data/mask/"
    data_loaders = {}
    for _, area in tqdm(enumerate(areas), desc="Processing areas"):
        path = data_dir + area + ".txt"
        data = {f"{area}": []}
        for line in open(path, "r"):
            data[f"{area}"].append(line.strip())
        dataset = Dataset.from_dict(data).map(
            lambda e: tokenizer(e[f"{area}"], truncation=True, padding='max_length', max_length=256), 
            batched=True,
            num_proc=4,
            remove_columns=area
        ).with_format("torch")
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collator)
        data_loaders[f"{area}"] = dataloader
    return data_loaders

if __name__ == "__main__":
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        wandb.init(
            project="train_mask_test_engineer",
            name=time.strftime("%Y-%m-%d--%H:%M")
        )
    path = "/home/qhn/Codes/Models/Bert-base-chinese"
    model, tokenizer = load_model(path)
    data_loaders = load_datasets(tokenizer)
    engineer_loader = data_loaders['工学']
    optimizer = AdamW(model.parameters(), lr=1e-5)
    for dataset_name, data_loader in data_loaders.items():
        model, data_loader, optimizer, engineer_loader = accelerator.prepare(model, data_loader, optimizer, engineer_loader)
        model.train()
        # train 
        for _ in range(5):
            for batch in tqdm(data_loader, desc="Training"):
                outputs = model(**batch)
                loss, logits = outputs.loss, outputs.logits
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({f"loss of {dataset_name}": loss})
        # eval
        model.eval()
        f1s, accs = [], []
        for batch in tqdm(engineer_loader, desc="Evaluating"):
            outputs = model(**batch)
            loss, logits = outputs.loss, outputs.logits
            if accelerator.is_main_process:
                true_labels = accelerator.gather(batch["labels"])
                pred_labels = accelerator.gather(logits.argmax(dim=-1))
                true_ids = true_labels[true_labels != -100].to("cpu").numpy()
                pred_ids = pred_labels[true_labels != -100].to("cpu").numpy()

                f1 = f1_score(true_ids, pred_ids, average="macro")
                acc = accuracy_score(true_ids, pred_ids)
            
                f1s.append(f1)
                accs.append(acc)
                wandb.log({f"f1 after {dataset_name} on engineer": f1, f"acc after {dataset_name} on engineer": acc})
        f1s, accs = np.array(f1s), np.array(accs)
        np.savetxt(f"/home/qhn/Codes/Projects/KnowledgeDist/Results/train_mask_eval_engineer/{dataset_name}f1.txt", f1s)
        np.savetxt(f"/home/qhn/Codes/Projects/KnowledgeDist/Results/train_mask_eval_engineer/{dataset_name}acc.txt", accs)
    print(len(data_loaders))