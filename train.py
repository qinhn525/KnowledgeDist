from transformers import (
    BertConfig, 
    AutoTokenizer, 
    BertForSequenceClassification, 
    AdamW,
    get_scheduler
)
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from config import parsing_finetune
from loguru import logger
import  json

def load_model():
    config = BertConfig.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese", num_labels=26)
    model = BertForSequenceClassification.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese", config=config)
    tokenizer = AutoTokenizer.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese")
    return model, tokenizer

def load_train_valid():
    data_paths = {
        "train": "/home/qhn/Codes/Projects/KnowledgeDist/Data/engineer_train_50_bs.tsv",
        "valid": "/home/qhn/Codes/Projects/KnowledgeDist/Data/engineer_test.tsv"
    }
    label2index = json.load(open("/home/qhn/Codes/Projects/KnowledgeDist/Data/label2index.json", "r"))
    train_data, test_data = {"text": [],"label": []}, {"text": [],"label": []}
    with open(data_paths["train"], "r") as f:
        for line in f.readlines():
            text, label = line.strip().split("\t")
            train_data['text'].append(text)
            train_data['label'].append(label2index[label])
    with open(data_paths["valid"], "r") as f:
        for line in f.readlines():
            text, label = line.strip().split("\t")
            test_data['text'].append(text)
            test_data['label'].append(label2index[label])
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "valid": Dataset.from_dict(test_data)
        }
    )

def main(args):
    accelerator = Accelerator()
    logger.add(
        args.log_path,
        encoding="utf-8",
        format="{level} | {time:YYYY-MM-DD HH:mm:ss} | {file} | {line} | {message}",
        rotation="500 MB"
    )
    model, tokenizer = load_model()
    datasets = load_train_valid()
    train_dataset = datasets['train']
    test_dataset = datasets['valid']

    test_dataset = test_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_loader = DataLoader(test_dataset, batch_size=130, shuffle=False, drop_last=False,
                             num_workers=8)
    train_dataset = train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(train_dataset, batch_size=130, shuffle=True, drop_last=False,
                              num_workers=8) #consider batch size
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                          weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps
    )
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    logger.info(f"accelerator.state: {accelerator.state}")

if __name__ == '__main__':
    args = parsing_finetune()
    main(args)