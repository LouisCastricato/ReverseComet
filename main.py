import json
from transformers import BartTokenizer, BartForConditionalGeneration
from seq2seq_trainer import *
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
import random
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import numpy.random

data = None

with open('data.json') as json_file: 
    data = json.load(json_file)

import functools
import operator
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)


special_tokens_dict = {'prompt' : '<pmpt>'}

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

class ReverseCometDataset(Dataset):
    def __init__(self, data, tokenizer):

        self.data = data
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Tokenize and cut off the last ten sentences
        txt = sent_tokenize(self.data[idx])[:-10]
        n_sent = len(txt)
        n_start = np.random.randint(low = min(10, n_sent), high = n_sent)
        n_length = np.random.randint(low = min(3, n_sent), high = min(15, n_sent))
        n_prompt = np.random.randint(low = 1, high = 3)

        exmpl = txt[n_start - n_length:n_start + 1]
        body = " ".join(exmpl[n_prompt:])
        prompt = " ".join(exmpl[:n_prompt])

        tgt = prompt
        src = body + " <pmpt> "
        '''
        start_ids = findOccurrences(src, "<")
        clause = list()
        for i,idx in enumerate(start_ids):
            if i == len(start_ids) - 1:
                clause.append(src[idx:])
                break
            clause.append(src[idx:start_ids[i+1] - 1])
        random.shuffle(clause)
        clause = list(map(lambda x: x + " ", clause))
        src = foldl(operator.add, "", clause)
        '''
        #text = self.tokenizer(self.data[self.keys[idx]], return_tensors="pt").to('cuda')
        #label = self.tokenizer(self.keys[idx], return_tensors="pt").to('cuda')
        sample = {'src_texts': src, 'tgt_texts': tgt}

        return sample
model_name = 'facebook/bart-base'

data = load_dataset("pg19", cache_dir = "cache/")

#Download models
tokenizer =  BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

#Add the tokens above
tokenizer.add_tokens(list(special_tokens_dict.values()))
model.resize_token_embeddings(len(tokenizer))

#Set up datasets
config = model.config
train_dataset = ReverseCometDataset(data['train']['text'], tokenizer)
eval_dataset = ReverseCometDataset(data['validation']['text'], tokenizer)
del data

training_args = Seq2SeqTrainingArguments()
#training_args.max_steps *= 3
training_args.per_device_train_batch_size = 10

data_args = DataTrainingArguments()
trainer = Seq2SeqTrainer(config=config, model=model, compute_metrics=None,\
    train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args, data_args=data_args,\
    data_collator=Seq2SeqDataCollator(tokenizer, data_args, 4))
trainer.args.max_steps *= 3
trainer.train(
    model_path="output.model"
)
trainer.save_model()
'''
for i in range(0, 35):
    print(list(data["dev"].keys())[i])
    print(list(data["dev"].values())[i])
    inp_text = list(data["dev"].keys())[i]
    #print(tokenizer.encode('<blank>', add_prefix_space=False))
    #print(tokenizer.encode(list(data["test"].keys())[i]))
    #sys.exit()
    inputs = tokenizer([inp_text], max_length=1024, return_tensors='pt').to('cuda')
    outputs = model.generate(inputs['input_ids'], num_beams=30, max_length=35,\
        early_stopping=True, repetition_penalty=2.0)

    print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
'''