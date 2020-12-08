from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import numpy as np
import csv
from tqdm import tqdm
import multiprocessing

data = load_dataset("pg19", cache_dir= "cache/")

train = list()
val = list()

def preproc(inp):
    txt = sent_tokenize(inp)[:-10]

    n_sent = len(txt)
    try:
        n_start = np.random.randint(low = min(10, n_sent), high = n_sent)
        n_length = np.random.randint(low = min(3, n_sent), high = min(15, n_sent))
        n_prompt = np.random.randint(low = 1, high = 3)

        exmpl = txt[n_start - n_length:n_start + 1]
        body = " ".join(exmpl[n_prompt:])
        prompt = " ".join(exmpl[:n_prompt])

        tgt = prompt
        src = body + " <pmpt> "
    except:
        tgt = "None"
        src = "None"
    return [src, tgt]

pool = multiprocessing.Pool(processes=8)
train = pool.map(preproc, data['train']['text'][0:100])
print(train)
sys.exit()

with open("train.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(train)

print("SAVED TRAINING SET")

val = pool.map(preproc, data['validation']['text'])


with open("validation.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(val)

print("SAVED EVAL SET")


pool.close()