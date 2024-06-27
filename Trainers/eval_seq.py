import string
from transformers import (
    BertConfig, 
    BertTokenizer, 
    BertForSequenceClassification, 
)
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd, torch, json, os, numpy as np

def load_model_eval(dataset_name: string, num_labels: int, bone):
    model_path = f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{bone}"
    classipath = f"/home/qhn/Codes/Projects/KnowledgeDist/Weights/bert_{dataset_name}"  + "/classifier.pth"
    data_paths = {
        "test": f"/home/qhn/Codes/Projects/KnowledgeDist/Data/{dataset_name}/test.tsv"
    }
    label2index = json.load(open(f"/home/qhn/Codes/Projects/KnowledgeDist/Data/{dataset_name}/label2index.json", "r"))
    logger.info(f"model_path:{model_path}, classipath:{classipath}, data_paths:{data_paths}")
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = num_labels
    model = BertForSequenceClassification(config)
    model.bert = model.bert.from_pretrained(model_path)
    model.classifier.load_state_dict(torch.load(classipath))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    test_data = {"text": [],"labels": []}
    test  = pd.read_csv(data_paths['test'], sep="\t")
    for idx, row in test.iterrows():
        text, label = row['abstract'], label2index[row['discipline']]
        test_data['text'].append(text)
        test_data['labels'].append(label)
    
    test_dataset = Dataset.from_dict(test_data)
    
    test_dataset = test_dataset.map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=256), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=0)
    model = model.cuda(1).eval()
    return model, test_loader
                 
if __name__ == '__main__':
    datasets, num_labels = ["agriculture", "medicine", "science"], [7, 5, 9]
    acc_path = "/home/qhn/Codes/Projects/KnowledgeDist/Results/eval_seq/acc_results"
    f1_path = "/home/qhn/Codes/Projects/KnowledgeDist/Results/eval_seq/f1_results"

    f1s = np.zeros((3, 3), dtype=np.float32)
    accs = np.zeros((3, 3), dtype=np.float32)
    
    for idx, (dataset, num_label) in enumerate(zip(datasets, num_labels)):
        logger.info(f"-------Test {idx}-------")
        models, datas = [], []
        for i in range(idx + 1):
            model, data = load_model_eval(datasets[i], num_labels[i], bone=dataset)
            models.append(model)
            datas.append(data)
        progress_bar_model = tqdm(range(len(models)), desc="model")
        for idx_data, (model, data) in enumerate(zip(models, datas)):
            label_list, pred_list = [], []
            progress_bar_test = tqdm(range(len(data)), desc="test")
            for _, test in enumerate(data):
                test["input_ids"] = test["input_ids"].to("cuda:1")
                test["attention_mask"] = test["attention_mask"].to("cuda:1")
                test["labels"] = test["labels"].to("cuda:1")
                res = model(**test)
                pred = res.logits.max(1)[1]
                
                predictions = pred.cpu().numpy().tolist()
                references = test['labels'].cpu().numpy().tolist()
                
                label_list += references
                pred_list += predictions
                progress_bar_test.update(1)
            macro_f1 = f1_score(label_list, pred_list, average='macro')
            accuracy = sum([float(label_list[i] == pred_list[i]) for i in range(len(label_list))]) * 1.0 / len(pred_list)
            
            f1s[idx][idx_data] = macro_f1
            accs[idx][idx_data] = accuracy
            progress_bar_model.update(1)
            logger.info(f"f1s:{f1s}")
        for i in range(len(models)):
            del models[0]
            del datas[0]
    np.savetxt(f1_path, f1s, '%.4f', delimiter='\t')
    np.savetxt(acc_path, accs, '%.4f', delimiter='\t')