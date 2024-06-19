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

def main(args):
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
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
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=130, shuffle=False, drop_last=False,
                             num_workers=0)
    train_dataset = train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(train_dataset, batch_size=130, shuffle=False, drop_last=False,
                              num_workers=0)
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
    
    progress_bar_train = tqdm(range(len(train_loader)), disable=not accelerator.is_local_main_process)
    
    steps = 0
    trained_data = []
    for _, batch in enumerate(train_loader):
        # 正向、反向传播
        model.train()
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        progress_bar_train.update(1)
        steps += 1
        if accelerator.is_main_process:
            logger.info(f"loss: {loss}, steps: {steps}")
        
        # 测试准确率和F1
        accelerator.wait_for_everyone()
        model.eval()
        label_list = []
        pred_list = []
        progress_bar_test = tqdm(range(len(test_loader)), disable=not accelerator.is_local_main_process)
        for _, valid in enumerate(test_loader):
            res = model(**valid)
            pred = res.logits.max(1)[1]
            
            predictions = accelerator.gather(pred).cpu().numpy().tolist()
            references = accelerator.gather(valid['labels']).cpu().numpy().tolist()
            
            label_list += references
            pred_list += predictions
            progress_bar_test.update(1)
        accelerator.wait_for_everyone()
        micro_f1 = f1_score(label_list, pred_list, average='micro')
        accuracy = sum([float(label_list[i] == pred_list[i]) for i in range(len(label_list))]) * 1.0 / len(pred_list)
        if accelerator.is_local_main_process:
            logger.info(f"batch {_}: micro_f1: {micro_f1}, accuracy: {accuracy}")
        

if __name__ == '__main__':
    args = parsing_finetune()
    main(args)