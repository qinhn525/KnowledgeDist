from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
import datasets

def load_model():
    model = AutoModel.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("/home/qhn/Codes/Models/Bert-base-chinese")
    return model, tokenizer

def load_dataset():
    
def main():
    