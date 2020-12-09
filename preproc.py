from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import numpy as np
import csv
from tqdm import tqdm
import multiprocessing
import pandas as pd

data = load_dataset("pg19", cache_dir= "cache/")

val = list()

def preproc(inp):
    try:
        txt = sent_tokenize(inp)[10:-30]
        n_sent = len(txt)
        n_start = np.random.randint(low = min(10, n_sent), high = n_sent)
        n_length = np.random.randint(low = min(3, n_sent), high = min(15, n_sent))
        n_prompt = np.random.randint(low = 1, high = 3)

        exmpl = txt[n_start - n_length:n_start + 1]
        body = " ".join(exmpl[n_prompt:]).replace("\n", " ")
        prompt = " ".join(exmpl[:n_prompt]).replace("\n", " ")

        tgt = prompt
        src = body + " <pmpt> "
    except:
        tgt = "None"
        src = "None"
    return [src, tgt]
#Double up the lists, since its randomly sampled we should get zero overlap
train_input = data['train']['text']
for i in range(13):
    train_input += data['train']['text']

pool = multiprocessing.Pool(processes=8)
train = pd.DataFrame(pool.map(preproc, train_input))
train.to_csv('train.csv', index=False, header=False)

print("SAVED TRAINING SET")

val_input = data['validation']['text']
for i in range(5):
    train_input += data['validation']['text']

val = pd.DataFrame(pool.map(preproc, val_input))
val.to_csv('validation.csv', index=False, header=False)

print("SAVED EVAL SET")


pool.close()
