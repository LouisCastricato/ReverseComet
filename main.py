import json
from transformers import BartTokenizer, BartForConditionalGeneration
from seq2seq_trainer import *
import torch
from torch.utils.data import Dataset, DataLoader
from config import *

data = None

with open('data.json') as json_file: 
    data = json.load(json_file)


special_tokens_dict = {'xAttr': '<xAttr>', 'xEffect': '<xEffect>', 'xIntent': '<xIntent>',\
    'xNeed' : '<xNeed>', 'xReact' : '<xReact>', 'xWant' : '<xWant>', 'oEffect' : '<oEffect>',\
    'oReact' : '<oReact>', 'oWant' : '<oWant>', 'Event' : '<Event>', 'personx' : 'PersonX',\
    'persony' : 'PersonY', 'personz' : 'PersonZ'}

class ReverseCometDataset(Dataset):
    def __init__(self, data, tokenizer):

        self.data = data
        self.keys = data.keys()
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.tokenizer(self.data[self.keys[idx]], return_tensors="pt")
        label = self.tokenizer(self.keys[idx], return_tensors="pt")
        sample = {'text': text, 'labels': label}

        return sample

model_name = 'facebook/bart-large'

#Download models
tokenizer =  BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

#Add the tokens above
tokenizer.add_tokens(special_tokens_dict.values())
model.resize_token_embeddings(len(tokenizer))

#Set up datasets
config = model.config
train_dataset = ReverseCometDataset(data["train"], tokenizer)
eval_dataset = ReverseCometDataset(data["dev"], tokenizer)
test_dataset = ReverseCometDataset(data["test"], tokenizer)

training_args = Seq2SeqTrainingArguments()
trainer = Seq2SeqTrainer(config=config, model=model, compute_metrics=None,\
    train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args)



inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)
logits = outputs[0]