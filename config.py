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

    label_smoothing = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler = field(
        default="linear"
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length  = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length  = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train  = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val  = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test  = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang  = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang  = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams  = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
