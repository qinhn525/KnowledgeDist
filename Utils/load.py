from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import json, pandas as pd, torch

def load_dataloader(dataset_name, tokenizer, args):
    data_paths = {
        "train": f"/home/qhn/Codes/Projects/KnowledgeDist/Data/{dataset_name}/train.tsv",
        "test": f"/home/qhn/Codes/Projects/KnowledgeDist/Data/{dataset_name}/test.tsv"
    }
    label2index = json.load(open(f"/home/qhn/Codes/Projects/KnowledgeDist/Data/{dataset_name}/label2index.json", "r"))
    train_data, test_data = {"text": [],"labels": []}, {"text": [],"labels": []}
    train = pd.read_csv(data_paths['train'], sep="\t")
    test  = pd.read_csv(data_paths['test'], sep="\t")
    for idx, row in train.iterrows():
        text, label = row['abstract'], label2index[row['discipline']]
        train_data['text'].append(text)
        train_data['labels'].append(label)
    for idx, row in train.iterrows():
        text, label = row['abstract'], label2index[row['discipline']]
        test_data['text'].append(text)
        test_data['labels'].append(label)
    
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    train_dataset = train_dataset.map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False,
                             num_workers=0)
    test_dataset = test_dataset.map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False,
                             num_workers=0)
    
    return train_loader, test_loader