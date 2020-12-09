import json
from transformers import BartTokenizer, BartForConditionalGeneration
from seq2seq_trainer import *
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
import random
from nltk.tokenize import sent_tokenize
import numpy.random
import csv
import argparse
import os

data = None

import functools
import operator
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)


special_tokens_dict = {'prompt' : '<pmpt>'}

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

class ReverseCometDataset(Dataset):
    def __init__(self, tokenizer, p_type = 'train'):
        with open(p_type + ".csv", "r") as f:
            self.data = list(csv.reader(f))

        self.p_type = p_type
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        tgt = self.data[idx][1]
        src = self.data[idx][0][:len(" <pmpt> ")]

        sample = {'src_texts': src, 'tgt_texts': tgt}

        return sample

#Model args, should enable fp16
parser = argparse.ArgumentParser()
parser.add_argument("--eval", action="store_true", default=False)
#If this is not bart-base or bart-large, assume load from file
parser.add_argument("--model", type=str, default="facebook/bart-base")
parser.add_argument("--eval_file", type=str, default="query.txt")
args = parser.parse_args()


model_name = args.model

#Download models
tokenizer =  BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(model_name)



#Add the tokens above
tokenizer.add_tokens(list(special_tokens_dict.values()))
model.resize_token_embeddings(len(tokenizer))
if not args.eval:
    #Set up datasets
    config = model.config
    train_dataset = ReverseCometDataset(tokenizer)
    eval_dataset = ReverseCometDataset(tokenizer, "validation")

    training_args = Seq2SeqTrainingArguments()
    #training_args.max_steps *= 3
    training_args.per_device_train_batch_size = 2
    training_args.fp16_opt_level = "O2"
    training_args.fp16 = True
    training_args.gradient_accumulation_steps = 3
    training_args.save_steps = 1000
    training_args.save_total_limit = 5

    data_args = DataTrainingArguments()
    trainer = Seq2SeqTrainer(config=config, model=model, compute_metrics=None,\
        train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args, data_args=data_args,\
        data_collator=Seq2SeqDataCollator(tokenizer, data_args, 4))
    trainer.args.fp16 = True
    trainer.train(
        model_path="output.model"
    )
    trainer.save_model()
else:
    f = open(args.eval_file, 'r')
    input_str = f.read()
    f.close()
    model.to("cuda:0")
    inputs = tokenizer([input_str], max_length=1024, return_tensors='pt').to('cuda:0')
    beam_outputs = model.generate(inputs['input_ids'], num_beams=5)
    print(tokenizer.decode(beam_outputs))
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