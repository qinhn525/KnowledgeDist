from transformers import (
    BertConfig, 
    AutoTokenizer, 
    BertModel, 
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
from  torch.nn import Linear
import json, numpy

class MyModel():
    def __init__(self, model_path:str, num_labels: int):
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.classifier = Linear(self.config['hidden_size'], num_labels)
    def forward(input):
        