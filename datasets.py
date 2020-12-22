from torch.utils.data import Dataset
import torch
import random
from nltk.tokenize import sent_tokenize
import numpy.random
import csv
import os
import argparse
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

data = None

import functools
import operator




foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)



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
        src = self.data[idx][0][:-len(" <pmpt> ")]

        sample = {'src_texts': src, 'tgt_texts': tgt}

        return sample


class NarrativeQASummariesDataset(Dataset):
  def __init__(self, summaries_file, map_file):

    # create df for both
    self.summaries_df = pd.read_csv(summaries_file)

    #Randomize order
    self.map_file = shuffle(pd.read_csv(map_file))
  def __len__(self):
    # return the len of the number of documents --> len of summaries_df
    return len(self.map_file)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    q_map = self.map_file.iloc[idx]

    summary_idx = q_map[0]
    question = q_map[1]
    answer = q_map[2]
    summary = self.summaries_df.iloc[summary_idx]['summary'].replace("\n", " ")
    answer_idx = np.random.randint(2)

    input_txt = "question: {} + context: {}".format(question, summary)

    #Remove paranthesis and split
    answer_out = answer[1:-1].split(", ")[answer_idx]
    #Remove quotations
    answer_out = answer_out[1:-1]
    
    sample = {'src_texts' : input_txt, 'tgt_texts' : answer_out}
    
    return sample