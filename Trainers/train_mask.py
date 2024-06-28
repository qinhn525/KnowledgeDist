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
from config import parsing_finetune
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import json, numpy as np, pandas as pd, wandb

def load_model(path):
    config = BertConfig.from_pretrained(path)
    model = BertForMaskedLM.from_pretrained(path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_dataset(tokenizer):
    data_path = "/home/qhn/Codes/Projects/KnowledgeDist/Data/csl_40k.tsv"
    data = pd.read_csv(data_path, sep="\t")[["abstract"]]
    data.columns = ["text"]
    dataset = Dataset.from_pandas(data).map(
        lambda e: tokenizer(e["text"], truncation=True, padding='max_length', max_length=256), 
        batched=True,
        num_proc=4,
        remove_columns="text"
    ).with_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    return dataset, collator

if __name__ == "__main__":
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        wandb.init(
            project="train_mask",
            name="train_mlm"
        )
    path = "/home/qhn/Codes/Models/Bert-base-chinese"
    model, tokenizer = load_model(path)
    dataset, collator = load_dataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    optimizer.zero_grad()
    for epoch in range(5):
        f1s, accs = [], []
        model.train()
        for batch in tqdm(dataloader):
            outputs = model(**batch)
            loss, logits = outputs.loss, outputs.logits
            
            
            if accelerator.is_local_main_process:
                true_labels = accelerator.gather(batch["labels"])
                pred_labels = accelerator.gather(logits.argmax(dim=-1))
                true_ids = true_labels[true_labels != -100].to("cpu").numpy()
                pred_ids = pred_labels[true_labels != -100].to("cpu").numpy()

                f1 = f1_score(true_ids, pred_ids, average="macro")
                acc = accuracy_score(true_ids, pred_ids)
            
                f1s.append(f1)
                accs.append(acc)
                wandb.log({"f1": f1, "acc": acc, "loss": loss})
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.is_local_main_process:
            np.savetxt(f"/home/qhn/Codes/Projects/KnowledgeDist/Results/mask/f1_epoch:{epoch}.txt", np.array(f1s))
            np.savetxt(f"/home/qhn/Codes/Projects/KnowledgeDist/Results/mask/acc_epoch:{epoch}.txt", np.array(accs))