import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from seq2seq_trainer import Seq2SeqTrainer, arg_to_scheduler_choices
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy
import numpy as np


def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    def __init__(self):
        self.label_smoothing = 0.0 
        self.sortish_sampler = False 
        self.predict_with_generate = False
        self.adafactor = False
        self.encoder_layerdrop = None
        self.decoder_layerdrop = None
        self.dropout = None
        self.attention_dropout = None
        self.lr_scheduler = "linear"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    def __init__(self):
        self.data_dir = ""
        self.task = "summarization"
        self.max_source_length = 1024
        self.max_target_length  = 128
        self.val_max_target_length  = 142
        self.test_max_target_length = 142
        self.n_train  = -1
        self.n_val  = -1
        self.n_test  = -1
        self.src_lang  = None
        self.tgt_lang  = None
        self.eval_beams  = None
