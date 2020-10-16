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
    'persony' : 'PersonY', 'personz' : 'PersonZ', 'blank' : '<blank>'}

class ReverseCometDataset(Dataset):
    def __init__(self, data, tokenizer):

        self.data = data
        self.keys = list(data.keys())
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #text = self.tokenizer(self.data[self.keys[idx]], return_tensors="pt").to('cuda')
        #label = self.tokenizer(self.keys[idx], return_tensors="pt").to('cuda')
        sample = {'src_texts': self.data[self.keys[idx]], 'tgt_texts': self.keys[idx]}

        return sample

model_name = 'facebook/bart-base'

#Download models
tokenizer =  BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained("./result/checkpoint-8500/")

#Add the tokens above
tokenizer.add_tokens(special_tokens_dict.values())
model.resize_token_embeddings(len(tokenizer))

#Set up datasets
config = model.config
train_dataset = ReverseCometDataset(data["train"], tokenizer)
eval_dataset = ReverseCometDataset(data["dev"], tokenizer)
test_dataset = ReverseCometDataset(data["test"], tokenizer)

training_args = Seq2SeqTrainingArguments()
training_args.per_device_train_batch_size = 8

data_args = DataTrainingArguments()
trainer = Seq2SeqTrainer(config=config, model=model, compute_metrics=None,\
    train_dataset=train_dataset, eval_dataset=eval_dataset, args=training_args, data_args=data_args,\
    data_collator=Seq2SeqDataCollator(tokenizer, data_args, 4))


'''
trainer.train(
    model_path="output.model"
)
trainer.save_model()
'''
for i in range(0, 35):
    print(list(data["test"].keys())[i])
    print(list(data["test"].values())[i])
    inp_text = list(data["test"].keys())[i]
    #print(tokenizer.encode("<blank>", add_prefix_space=True))

    inputs = tokenizer([inp_text], max_length=1024, return_tensors='pt').to('cuda')
    outputs = model.generate(inputs['input_ids'], num_beams=20, max_length=35,\
        early_stopping=True, repetition_penalty=2.0, bad_words_ids=[len(tokenizer)-1],\
        no_repeat_ngram_size=4)
    print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
