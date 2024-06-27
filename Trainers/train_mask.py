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
from tqdm import tqdm
from sklearn.metrics import f1_score
import json, numpy

def load_model():
    config = BertConfig.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese", num_labels=26)
    model = BertForSequenceClassification.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese", config=config)
    tokenizer = AutoTokenizer.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese")
    return model, tokenizer

def load_dataset():
    data_paths = {
        "train": "/home/qhn/Codes/Projects/KnowledgeDist/Data/csl_40k.tsv"
    }
    label2index = json.load(open("/home/qhn/Codes/Projects/KnowledgeDist/Data/engineer/label2index.json", "r"))
    train_data, test_data = {"text": [],"labels": []}, {"text": [],"labels": []}
    with open(data_paths["train"], "r") as f:
        for line in f.readlines():
            text, label = line.strip().split("\t")
            train_data['text'].append(text)
            train_data['labels'].append(label2index[label])
    with open(data_paths["valid"], "r") as f:
        for line in f.readlines():
            text, label = line.strip().split("\t")
            test_data['text'].append(text)
            test_data['labels'].append(label2index[label])
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "valid": Dataset.from_dict(test_data)
        }
    )