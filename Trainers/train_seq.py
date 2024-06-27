from transformers import (
    BertConfig, 
    AutoTokenizer, 
    BertForSequenceClassification, 
    AdamW,
)
from accelerate import Accelerator
from config import parsing_finetune
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import f1_score
from Utils.load import load_dataloader
import pandas as pd, torch

def train(args):
    datasets, num_labels = ["agriculture", "medicine", "science"], [7, 5, 9]
    bert_base_dir = "/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{}"
    accelerator = Accelerator()
    results = {
        "dataset": [],
        "acc": [],
        "macro_f1": []
    }
    for idx, (dataset, num_label) in enumerate(zip(datasets, num_labels)):
        # 根据数据集读取对应权重，第一个任务初始权重为原生bert
        config = BertConfig.from_pretrained(bert_base_dir.format(dataset), num_labels=num_label)
        model = BertForSequenceClassification.from_pretrained(bert_base_dir.format(dataset), config=config)
        tokenizer = AutoTokenizer.from_pretrained(bert_base_dir.format(dataset))
        
        train_loader, test_loader = load_dataloader(dataset, tokenizer, args)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                          weight_decay=args.weight_decay)
        model, train_loader, test_loader, optimizer = accelerator.prepare(model, train_loader, test_loader, optimizer)
        
        progress_bar_train = tqdm(range(len(train_loader) * args.epoch), disable=not accelerator.is_local_main_process, desc="train")
        steps = 0
        model.train()
        for i in range(args.epoch):
            for _, batch in enumerate(train_loader):
                # 正向、反向传播
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # 更新参数
                accelerator.wait_for_everyone()
                optimizer.step()
                progress_bar_train.update(1)
                steps += 1
                if accelerator.is_main_process:
                    logger.info(f"epoch:{i} -> loss: {loss}, steps: {steps}")
        # 测试准确率和F1  
        accelerator.wait_for_everyone()
        model.eval()
        label_list = []
        pred_list = []
        progress_bar_test = tqdm(range(len(test_loader)), disable=not accelerator.is_local_main_process, desc="test")
        for _, test in enumerate(test_loader):
            res = model(**test)
            pred = res.logits.max(1)[1]
            
            predictions = accelerator.gather(pred).cpu().numpy().tolist()
            references = accelerator.gather(test['labels']).cpu().numpy().tolist()
            
            label_list += references
            pred_list += predictions
            progress_bar_test.update(1)
        accelerator.wait_for_everyone()
        macro_f1 = f1_score(label_list, pred_list, average='macro')
        accuracy = sum([float(label_list[i] == pred_list[i]) for i in range(len(label_list))]) * 1.0 / len(pred_list)
        
        # 每次训练后保存结果到字典中，三个都训完后保存
        results['dataset'].append(dataset)
        results['acc'].append(accuracy)
        results['macro_f1'].append(macro_f1)
        
        model = accelerator.unwrap_model(model)
        
        # 如果不是最后一个数据集，则模型在下一任务的模型文件下也保存，这样下一任务开始训练时，读取的权重就是上一任务训练好的权重
        if idx != 2:
            model.bert.save_pretrained(f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{datasets[idx + 1]}", safe_serialization=False)
            tokenizer.save_pretrained(f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{datasets[idx + 1]}")
        tokenizer.save_pretrained(f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{datasets[idx]}")
        model.bert.save_pretrained(f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{datasets[idx]}", safe_serialization=False)
        torch.save(model.classifier.state_dict(), f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{datasets[idx]}/classifier.pth")
        
        if accelerator.is_local_main_process:
            logger.info(f"test_dataset {dataset}: macro_f1: {macro_f1}, accuracy: {accuracy}")
        del model
    
    # 保存训练结果
    df = pd.DataFrame(results)
    df.to_csv("/home/qhn/Codes/Projects/KnowledgeDist/Results/seq/results.tsv", sep="\t", index=False)
if __name__ == "__main__":
    args = parsing_finetune()
    train(args)