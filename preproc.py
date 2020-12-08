from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import numpy as np
import csv
from tqdm import tqdm

data = load_dataset("pg19", cache_dir= "cache/")

train = list()
val = list()

for book in tqdm(data['train']['text']):
    txt = sent_tokenize(book)[:-10]

    n_sent = len(txt)
    n_start = np.random.randint(low = min(10, n_sent), high = n_sent)
    n_length = np.random.randint(low = min(3, n_sent), high = min(15, n_sent))
    n_prompt = np.random.randint(low = 1, high = 3)

    exmpl = txt[n_start - n_length:n_start + 1]
    body = " ".join(exmpl[n_prompt:])
    prompt = " ".join(exmpl[:n_prompt])

    tgt = prompt
    src = body + " <pmpt> "
    train.append([src, tgt])

with open("train.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(train)

print("SAVED TRAINING SET")

for book in tqdm(data['train']['text']):
    txt = sent_tokenize(book)[:-10]

    n_sent = len(txt)
    n_start = np.random.randint(low = min(10, n_sent), high = n_sent)
    n_length = np.random.randint(low = min(3, n_sent), high = min(15, n_sent))
    n_prompt = np.random.randint(low = 1, high = 3)

    exmpl = txt[n_start - n_length:n_start + 1]
    body = " ".join(exmpl[n_prompt:])
    prompt = " ".join(exmpl[:n_prompt])

    tgt = prompt
    src = body + " <pmpt> "
    train.append([src, tgt])

with open("eval.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(train)
    
print("SAVED EVAL")