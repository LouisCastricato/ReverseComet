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
        txt = sent_tokenize(inp)[30:-30]
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

pool = multiprocessing.Pool(processes=8)
train = pd.DataFrame(pool.map(preproc, data['train']['text']))
train.to_csv('train.csv', index=False, header=False)

print("SAVED TRAINING SET")

val = pd.DataFrame(pool.map(preproc, data['validation']['text']))
val.to_csv('validation.csv', index=False, header=False)

print("SAVED EVAL SET")


pool.close()